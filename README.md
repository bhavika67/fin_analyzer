# FinAnalyzer — Intelligent Financial Document Analyzer & Q&A Agent

![CI/CD](https://github.com/bhavika67/fin-analyzer/actions/workflows/ci.yml/badge.svg)
![RAG Eval](https://github.com/bhavika67/fin-analyzer/actions/workflows/eval.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)


An end-to-end **Agentic RAG system** for financial document intelligence. Ingests PDFs, DOCX, CSV, and Excel files — runs EDA, regression, and anomaly detection — and answers natural language questions using a LangGraph ReAct agent backed by a FAISS vector store and SQLite database.

---

## Architecture

```
data/raw/          ← PDFs, CSVs, DOCX, Excel files
     ↓
ingestion/         ← Parse → Chunk (1000 chars, 150 overlap)
     ↓
vectorstore/       ← OpenAI embeddings → FAISS index (842 vectors)
     ↓
agent/             ← LangGraph ReAct agent
                      Tool 1: SearchFinancialDocs   (vector retrieval)
                      Tool 2: SummarizeFinancialTopic (multi-doc synthesis)
                      Tool 3: QueryFinancialDatabase  (SQLite SQL queries)
                      Tool 4: GenerateChart           (matplotlib dark charts)
     ↓
api/               ← FastAPI backend
     ↓
ui/                ← Gradio dashboard (Ingest / Ask / EDA / Regression / Data)
```

---

## Features

| Feature | Description |
|---|---|
| **Document Ingestion** | Parse PDF, DOCX, CSV, Excel, TXT into semantic chunks |
| **Vector Store** | FAISS IndexFlatIP with OpenAI `text-embedding-3-small` |
| **Agentic RAG** | LangGraph ReAct agent with 4 specialized tools |
| **SQL Tool** | Agent queries SQLite directly for precise numeric answers |
| **Chart Tool** | Agent generates dark-themed matplotlib charts on demand |
| **EDA** | Statistical insights, trend analysis, correlation detection |
| **Regression** | Linear regression with R² gauge and coefficient visualization |
| **Anomaly Detection** | IQR and Z-score outlier detection |
| **RAGAS Evaluation** | Faithfulness, answer relevancy, context precision scoring |
| **FastAPI Backend** | REST API with `/ingest`, `/ask`, `/eda`, `/regression`, `/analyze` |
| **API Security** | API key auth, rate limiting, CORS lockdown, upload size guard |
| **Gradio UI** | 5-tab interactive dashboard with Plotly charts |
| **CI/CD Pipeline** | GitHub Actions — lint → test (3.10–3.12) → security scan → Docker build → deploy |
| **Containerized** | Multi-stage Dockerfile + Docker Compose with named volumes |
| **Weekly Eval** | Automated RAGAS evaluation every Sunday via GitHub Actions |

---

## Quickstart

### Windows

```powershell
# 1. Clone and move to a short path (avoids Windows WinError 206 path length errors)
git clone https://github.com/bhavika67/fin-analyzer.git
mkdir C:\fin
Copy-Item -Path "fin-analyzer\*" -Destination "C:\fin\" -Recurse
cd C:\fin

# 2. Enable long paths (run once as Administrator)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# 3. Install Poetry and dependencies
pip install poetry
python -m poetry config virtualenvs.path "C:\venvs"
python -m poetry install

# 4. Set up environment
copy .env.example .env
# Edit .env — add your OPENAI_API_KEY and API_SECRET_KEY
```

> **Note:** If `poetry` command is not recognised, use `python -m poetry` instead.

### Mac / Linux

```bash
# 1. Clone the repo
git clone https://github.com/bhavika67/fin-analyzer.git
cd fin-analyzer

# 2. Install Poetry and dependencies
curl -sSL https://install.python-poetry.org | python3 -
poetry install

# 3. Set up environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY and API_SECRET_KEY
```

### Environment variables

```env
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
VECTOR_STORE=faiss
FAISS_INDEX_PATH=data/embeddings/faiss_index
DATABASE_URL=sqlite:///data/processed/fin.db
LOG_LEVEL=INFO

# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
API_SECRET_KEY=your-secret-key-here
```

### Bootstrap data

```bash
# Generate synthetic financial data
poetry run python scripts/generate_sample_data.py

# Fetch real data from Yahoo Finance
poetry run python scripts/fetch_real_data.py --ticker AAPL --years 5
poetry run python scripts/fetch_real_data.py --ticker MSFT --years 5
poetry run python scripts/fetch_real_data.py --ticker GOOGL --years 5
poetry run python scripts/fetch_real_data.py --ticker TSLA --years 5
poetry run python scripts/fetch_real_data.py --ticker INFY.NS --years 5
poetry run python scripts/fetch_real_data.py --ticker TCS.NS --years 5

# Ingest into vector store
poetry run python scripts/ingest_all.py
# Expected: ~842 vectors indexed across 23 files

# Load into SQLite database
poetry run python scripts/load_sql_db.py
# Creates data/processed/fin.db with 9 tables
```

### Run the app

```bash
# Terminal 1 — API
poetry run uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — UI
poetry run python ui/app.py
```

Open **http://127.0.0.1:7860** in your browser.

---

## Project Structure

```
fin-analyzer/
├── config.py                    ← Pydantic settings (reads .env)
├── requirements.txt             ← Direct dependencies
├── requirements.lock            ← Pinned dependency versions (pip-compile)
├── Dockerfile                   ← Multi-stage build, non-root user
├── docker-compose.yml           ← api + ui + one-shot setup service
├── ruff.toml                    ← Linter + formatter config
├── pytest.ini                   ← Test config + markers
├── .dockerignore                ← Keeps image lean
├── .env.example                 ← Environment template (safe to commit)
│
├── .github/
│   ├── CODEOWNERS               ← Auto-assign reviewers
│   ├── dependabot.yml           ← Weekly dependency updates
│   ├── pull_request_template.md ← PR checklist
│   └── workflows/
│       ├── ci.yml               ← lint → test → security → build → deploy
│       └── eval.yml             ← Weekly RAGAS evaluation
│
├── ingestion/
│   ├── parser.py                ← PDF/DOCX/CSV/TXT parser
│   ├── chunker.py               ← RecursiveCharacterTextSplitter
│   └── pipeline.py              ← Orchestrates parse → chunk
│
├── eda/
│   ├── analyzer.py              ← EDA: summary stats, trends, correlations
│   ├── regression.py            ← Linear regression with StandardScaler
│   └── anomaly.py               ← IQR + Z-score anomaly detection
│
├── vectorstore/
│   ├── embedder.py              ← OpenAI text-embedding-3-small
│   └── store.py                 ← FAISS IndexFlatIP with L2 normalization
│
├── agent/
│   └── agent.py                 ← LangGraph ReAct agent + 4 tools
│
├── evaluation/
│   └── evaluator.py             ← RAGAS evaluation pipeline
│
├── api/
│   └── main.py                  ← FastAPI: auth + rate limiting + all routes
│
├── ui/
│   ├── app.py                   ← Gradio 5-tab dashboard
│   └── charts.py                ← Plotly dark-theme chart builders
│
├── scripts/
│   ├── generate_sample_data.py  ← Creates synthetic P&L, segment, headcount CSVs
│   ├── fetch_real_data.py       ← Yahoo Finance data fetcher
│   ├── ingest_all.py            ← Embeds and indexes data/raw/ into FAISS
│   ├── load_sql_db.py           ← Loads CSVs into SQLite
│   ├── ask.py                   ← CLI agent tester
│   ├── evaluate_rag.py          ← Runs RAGAS on 8 Q&A samples
│   └── check_db.py              ← Verifies SQLite tables and data
│
├── tests/
│   ├── conftest.py              ← Shared pytest fixtures
│   ├── test_ingestion.py        ← Parser + chunker tests (6 tests)
│   ├── test_eda.py              ← EDA + regression + anomaly tests (14 tests)
│   └── test_vectorstore.py      ← FAISS add/search/save/load tests (9 tests)
│
├── data/
│   ├── raw/                     ← Source files (CSVs, TXTs, PDFs) — gitignored
│   ├── processed/               ← SQLite database (fin.db) — gitignored
│   └── embeddings/              ← FAISS index files — gitignored
│
└── reports/
    └── output/                  ← RAGAS eval results — gitignored
```

---

## SQL Database Schema

The agent can query these tables directly using natural language:

```sql
quarterly_pl       (quarter, revenue, cogs, opex, ebitda, tax, net_profit, net_margin)
segment_revenue    (month, segment, revenue, customers, churn_rate)
cost_headcount     (month, headcount, salary_cost, infra_cost, marketing, rd_spend, total_cost)
aapl_financials    (period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
googl_financials   (period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
msft_financials    (period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
tsla_financials    (period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
infy_financials    (period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
tcs_financials     (period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
```

---

## API Endpoints

All `POST` endpoints require the `X-API-Key` header. `GET` endpoints are public.

| Method | Endpoint | Auth | Rate Limit | Description |
|---|---|---|---|---|
| `GET` | `/health` | ❌ | — | API status + vector count |
| `GET` | `/tables` | ❌ | — | List loaded database tables |
| `POST` | `/ingest` | ✅ | 20/min | Upload and index a document |
| `POST` | `/ask` | ✅ | 10/min | Natural language Q&A via agent |
| `POST` | `/eda` | ✅ | 20/min | EDA on uploaded CSV/Excel or table |
| `POST` | `/regression` | ✅ | 20/min | Linear regression on data |
| `POST` | `/analyze` | ✅ | 10/min | Full pipeline: ingest + EDA + regression |

Interactive docs available at **http://127.0.0.1:8000/docs**

**Example authenticated request:**
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the average net profit?"}'
```

---

## Example Questions

**Numeric (SQL tool):**
```
What is the average net profit across all quarters?
Which quarter had the highest revenue?
Compare AAPL and MSFT net income over the last 5 quarters
What is TSLA's gross margin trend?
```

**Visual (Chart tool):**
```
Show me a line chart of revenue from quarterly_pl
Plot AAPL net income as a bar chart
Visualize total cost trend from cost_headcount
```

**Qualitative (Vector search):**
```
What were the key risks mentioned in the annual report?
What caused the Q3 2022 revenue dip?
What is the FY2024 revenue outlook?
```

---

## Evaluation Results (RAGAS)

Evaluated on 8 financial Q&A samples after adding SQL tool:

| Metric | Before SQL Tool | After SQL Tool |
|---|---|---|
| Avg Faithfulness | 0.479 | **0.753** |
| Avg Answer Relevancy | 0.525 | **0.889** |
| Avg Context Precision | 0.625 | 0.625 |
| **Overall Score** | 0.543 (Fair) | **0.756 (Good)** |

The improvement was achieved by routing numeric questions through the SQL tool rather than relying on vector retrieval for structured financial data.

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v
# Expected: 29 passed

# Run with coverage report
python -m pytest tests/ -v --cov=. --cov-report=term-missing

# Run only fast unit tests
python -m pytest tests/ -v -m unit
```

---

## Docker

```bash
cp .env.example .env
# Add your OPENAI_API_KEY and API_SECRET_KEY to .env

# First time only — generate sample data and ingest
docker compose --profile setup up setup

# Start the full stack
docker compose up --build
# UI  → http://localhost:7860
# API → http://localhost:8000/docs
```

Data is persisted in named Docker volumes (`faiss_data`, `db_data`, `raw_data`) so it survives container restarts.

---

## CI/CD Pipeline

Every push to `main` or pull request triggers:

```
Lint (Ruff) → Tests (Py 3.10/3.11/3.12) → Security Scan (Bandit) → Docker Build → Deploy to Staging
```

| Job | Tool | What it checks |
|---|---|---|
| Lint | Ruff | Code style + import order |
| Tests | Pytest + coverage | 29 unit tests across 3 Python versions |
| Security | Bandit | Common security vulnerabilities |
| Build | Docker Buildx | Image builds cleanly, pushed to GHCR |
| Deploy | SSH | Auto-deploys to staging on `main` push |

**Weekly:** A separate workflow runs RAGAS evaluation every Sunday and uploads results as a GitHub Actions artifact.

**GitHub Secrets required:**

| Secret | Used by |
|---|---|
| `OPENAI_API_KEY` | eval workflow |
| `API_SECRET_KEY` | staging deploy |
| `STAGING_HOST` / `STAGING_USER` / `STAGING_SSH_KEY` | deploy job |

---



- **No PDF table extraction** — scanned PDFs and tables inside PDFs need `pdfplumber`
- **Fixed top-k retrieval** — always retrieves 4 chunks regardless of question complexity
- **No conversation memory** — each session starts fresh, no cross-session persistence
- **Linear regression only** — no non-linear models or time-series decomposition
- **SQLite** — single-writer, not suitable for concurrent production traffic
- **No streaming** — agent responses appear after full completion (5–15 seconds)
- **FAISS flat index** — exact search, suitable up to ~100K vectors before needing HNSW

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Agent framework | LangGraph (ReAct pattern) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | FAISS (IndexFlatIP, cosine similarity) |
| Database | SQLite (dev) |
| EDA / ML | Pandas, NumPy, Scikit-learn, SciPy |
| Evaluation | RAGAS |
| API | FastAPI + Uvicorn |
| API Security | API key auth, slowapi rate limiting, CORS, upload guard |
| UI | Gradio 6 |
| Visualization | Plotly, Matplotlib |
| Testing | Pytest (29 tests) + coverage |
| Linting | Ruff |
| Security scanning | Bandit |
| Containers | Docker (multi-stage), Docker Compose |
| CI/CD | GitHub Actions |
| Dependency locking | pip-tools (requirements.lock) |

---

## Author

**Bhavika Sharma** — Data Science & LLM Developer  
[LinkedIn](http://www.linkedin.com/in/bhavika-sharma-ml) · [GitHub](https://github.com/bhavika67/) · [Hugging Face](https://huggingface.co/Bhavika67)
