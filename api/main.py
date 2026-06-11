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
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from config import get_settings
from ingestion.pipeline import IngestionPipeline
from vectorstore.store import VectorStore
from agent.agent import FinancialAgent
from eda import EDAAnalyzer, RegressionAnalyzer, AnomalyDetector

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Financial Document Analyzer",
    description="Ingest financial documents and ask questions via Agentic RAG.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (loaded once at startup) ───────────────────────────────────────
settings        = get_settings()
vector_store    = VectorStore()
vector_store.load()
ingestion       = IngestionPipeline()
agent           = FinancialAgent(vector_store)
eda_analyzer    = EDAAnalyzer()
reg_analyzer    = RegressionAnalyzer()
anomaly_detector = AnomalyDetector()

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


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
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
def ask(request: QuestionRequest):
    """Ask a natural language question about ingested documents."""
    result = agent.ask(request.question)
    return QuestionResponse(**result)


@app.post("/eda")
async def run_eda(file: UploadFile = File(...), target_column: str | None = None):
    """Run EDA on an uploaded CSV or Excel file."""
    import pandas as pd
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        df = pd.read_csv(tmp_path) if suffix == ".csv" else pd.read_excel(tmp_path)
        result   = eda_analyzer.analyze(df, target_column)
        anomalies = anomaly_detector.detect(df)
        return {
            "shape":         {"rows": len(df), "cols": len(df.columns)},
            "insights":      result.insights,
            "trends":        result.trends,
            "correlations":  result.correlations,
            "missing_values": result.missing_values,
            "anomalies":     [{"column": a.column, "count": a.anomaly_count,
                               "summary": a.summary} for a in anomalies],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/regression")
async def run_regression(file: UploadFile = File(...), target_column: str = "revenue"):
    """Run linear regression on an uploaded CSV."""
    import pandas as pd
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        df     = pd.read_csv(tmp_path) if suffix == ".csv" else pd.read_excel(tmp_path)
        result = reg_analyzer.fit(df, target_column)
        return {
            "target":         result.target,
            "r2":             result.r2,
            "mae":            result.mae,
            "rmse":           result.rmse,
            "interpretation": result.interpretation,
            "coefficients":   result.coefficients,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)