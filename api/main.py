# api/main.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

import tempfile
import shutil
import sqlite3
from fastapi import FastAPI, UploadFile, File, HTTPException, Security, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from loguru import logger

from config import get_settings
from ingestion.pipeline import IngestionPipeline
from vectorstore.store import VectorStore
from agent.agent import FinancialAgent, DB_PATH
from eda import EDAAnalyzer, RegressionAnalyzer, AnomalyDetector

# ── App setup ─────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Financial Document Analyzer",
    description="Ingest financial documents and ask questions via Agentic RAG.",
    version="1.0.0",
)

# ── 1. Rate limiting ───────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── 2. CORS — only allow the Gradio UI origin ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:7860",
        "http://localhost:7860",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# ── 3. Trusted host ────────────────────────────────────────────────────────────
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost"],
)

# ── 4. Upload size limit (50 MB) ───────────────────────────────────────────────
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

class LimitUploadSize(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "File too large. Maximum upload size is 50 MB."},
                )
        return await call_next(request)

app.add_middleware(LimitUploadSize)

# ── Singletons (loaded once at startup) ───────────────────────────────────────
settings        = get_settings()
vector_store    = VectorStore()
vector_store.load()
ingestion       = IngestionPipeline()
agent           = FinancialAgent(vector_store)
eda_analyzer    = EDAAnalyzer()
reg_analyzer    = RegressionAnalyzer()
anomaly_detector = AnomalyDetector()

# Tables the SQL agent tool knows about — same whitelist used to gate
# EDA/regression-on-table requests so an arbitrary string can't be interpolated
# into a SQL query.
ALLOWED_TABLES = {
    "quarterly_pl", "segment_revenue", "cost_headcount",
    "aapl_financials", "googl_financials", "msft_financials",
    "tsla_financials", "infy_financials", "tcs_financials",
}

# ── API key authentication ─────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(_api_key_header)):
    if not settings.api_secret_key:
        return  # auth disabled if key not configured
    if key != settings.api_secret_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Pass it as X-API-Key header.",
        )


def _load_table(table_name: str):
    """Load a whitelisted table from the financial database as a DataFrame."""
    import pandas as pd
    if table_name not in ALLOWED_TABLES:
        raise HTTPException(status_code=400, detail=f"Unknown table '{table_name}'.")
    if not DB_PATH.exists():
        raise HTTPException(status_code=400,
                             detail="Database not found. Run scripts/load_sql_db.py first.")
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()


def _load_dataframe(file: UploadFile | None, table_name: str | None):
    """Resolve a DataFrame from either an uploaded file or a database table.
    Writes the data to a temp directory under a meaningful name — the
    original upload filename, or '<table_name>.csv' for tables — so that if
    the caller goes on to ingest this same path (as /analyze does), the
    vector store ends up with accurate source metadata instead of a random
    temp filename. Returns (df, label, tmp_path, tmp_dir); tmp_dir must be
    rmtree'd by the caller."""
    import pandas as pd
    if table_name:
        df = _load_table(table_name)
        tmp_dir  = tempfile.mkdtemp()
        tmp_path = str(Path(tmp_dir) / f"{table_name}.csv")
        df.to_csv(tmp_path, index=False)
        return df, table_name, tmp_path, tmp_dir
    if file is not None:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".csv", ".xlsx", ".xls"}:
            raise HTTPException(status_code=400,
                                 detail=f"Only CSV or Excel files are supported here "
                                        f"(got '{suffix}').")
        safe_name = Path(file.filename).name or f"upload{suffix}"
        tmp_dir   = tempfile.mkdtemp()
        tmp_path  = str(Path(tmp_dir) / safe_name)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        df = pd.read_csv(tmp_path) if suffix == ".csv" else pd.read_excel(tmp_path)
        return df, file.filename, tmp_path, tmp_dir
    raise HTTPException(status_code=400, detail="Provide either a file or a table_name.")


# ── Request / Response models ─────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str | None
    error:  str | None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "vectors_indexed": vector_store.total}


@app.get("/tables")
def list_tables():
    """List whitelisted tables currently loaded in the financial database."""
    if not DB_PATH.exists():
        return {"tables": []}
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        present = {row[0] for row in cursor.fetchall()}
    finally:
        conn.close()
    return {"tables": sorted(present & ALLOWED_TABLES)}


@app.post("/ingest")
@limiter.limit("20/minute")
async def ingest(request: Request, file: UploadFile = File(...), _=Security(verify_api_key)):
    """Upload and index a financial document."""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        chunks = ingestion.run(tmp_path)
        vector_store.add_chunks(chunks)
        vector_store.save()
        logger.info(f"Ingested {file.filename}: {len(chunks)} chunks")
        return {"filename": file.filename, "chunks_indexed": len(chunks),
                "total_vectors": vector_store.total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ask", response_model=QuestionResponse)
@limiter.limit("10/minute")
def ask(request: Request, body: QuestionRequest, _=Security(verify_api_key)):
    """Ask a natural language question about ingested documents."""
    result = agent.ask(body.question)
    return QuestionResponse(**result)


@app.post("/eda")
@limiter.limit("20/minute")
async def run_eda(request: Request,
                   file: UploadFile | None = File(None),
                   table_name: str | None = None,
                   target_column: str | None = None,
                   _=Security(verify_api_key)):
    """Run EDA on an uploaded CSV/Excel file, or on a table from the
    financial database (pass table_name instead of a file)."""
    tmp_dir = None
    try:
        df, label, tmp_path, tmp_dir = _load_dataframe(file, table_name)
        result    = eda_analyzer.analyze(df, target_column)
        anomalies = anomaly_detector.detect(df)
        return {
            "source":        label,
            "shape":         {"rows": len(df), "cols": len(df.columns)},
            "columns":       list(df.select_dtypes(include="number").columns),
            "insights":      result.insights,
            "trends":        result.trends,
            "correlations":  result.correlations,
            "missing_values": result.missing_values,
            "anomalies":     [{"column": a.column, "count": a.anomaly_count,
                               "summary": a.summary} for a in anomalies],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/regression")
@limiter.limit("20/minute")
async def run_regression(request: Request,
                          file: UploadFile | None = File(None),
                          table_name: str | None = None,
                          target_column: str = "revenue",
                          _=Security(verify_api_key)):
    """Run linear regression on an uploaded CSV/Excel file, or on a table
    from the financial database (pass table_name instead of a file)."""
    tmp_dir = None
    try:
        df, label, tmp_path, tmp_dir = _load_dataframe(file, table_name)
        result = reg_analyzer.fit(df, target_column)
        return {
            "source":         label,
            "target":         result.target,
            "r2":             result.r2,
            "mae":            result.mae,
            "rmse":           result.rmse,
            "interpretation": result.interpretation,
            "coefficients":   result.coefficients,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze(request: Request,
                   file: UploadFile | None = File(None),
                   table_name: str | None = None,
                   target_column: str | None = None,
                   _=Security(verify_api_key)):
    """Full pipeline: ingest a CSV/Excel dataset into the knowledge base, then
    run EDA on it (and regression too, if target_column is given). Accepts
    either an uploaded file or a table_name from the financial database.
    Unlike /eda and /regression, this also adds the data to the vector store —
    use those instead for one-off analysis you don't want persisted."""
    tmp_dir = None
    try:
        df, label, tmp_path, tmp_dir = _load_dataframe(file, table_name)

        # ── 1. Ingest into the knowledge base ──────────────────────────
        chunks = ingestion.run(tmp_path)
        vector_store.add_chunks(chunks)
        vector_store.save()
        logger.info(f"Ingested {label}: {len(chunks)} chunks")

        # ── 2. EDA ────────────────────────────────────────────────────
        eda_result = eda_analyzer.analyze(df, target_column)
        anomalies  = anomaly_detector.detect(df)

        response = {
            "source":         label,
            "chunks_indexed": len(chunks),
            "total_vectors":  vector_store.total,
            "shape":          {"rows": len(df), "cols": len(df.columns)},
            "columns":        list(df.select_dtypes(include="number").columns),
            "insights":       eda_result.insights,
            "trends":         eda_result.trends,
            "correlations":   eda_result.correlations,
            "missing_values": eda_result.missing_values,
            "anomalies":      [{"column": a.column, "count": a.anomaly_count,
                                "summary": a.summary} for a in anomalies],
            "regression":       None,
            "regression_error": None,
        }

        # ── 3. Regression (only if a target column was given) ───────────
        if target_column:
            try:
                reg_result = reg_analyzer.fit(df, target_column)
                response["regression"] = {
                    "target":         reg_result.target,
                    "r2":             reg_result.r2,
                    "mae":            reg_result.mae,
                    "rmse":           reg_result.rmse,
                    "interpretation": reg_result.interpretation,
                    "coefficients":   reg_result.coefficients,
                }
            except ValueError as e:
                response["regression_error"] = str(e)

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)