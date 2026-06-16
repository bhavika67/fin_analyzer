# FinAnalyzer — Intelligent Financial Document Analyzer & Q&A Agent

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
| **Gradio UI** | 5-tab interactive dashboard with Plotly charts |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/bhavika67/fin-analyzer.git
cd fin-analyzer
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

```env
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
VECTOR_STORE=faiss
FAISS_INDEX_PATH=data/embeddings/faiss_index
DATABASE_URL=sqlite:///data/processed/fin.db
LOG_LEVEL=INFO
```

### 3. Generate sample data + fetch real stock data

```bash
# Generate synthetic financial data (P&L, segment revenue, headcount)
python scripts/generate_sample_data.py

# Fetch real data from Yahoo Finance (AAPL, MSFT, GOOGL, TSLA, INFY, TCS)
python scripts/fetch_real_data.py --ticker AAPL --years 5
python scripts/fetch_real_data.py --ticker MSFT --years 5
python scripts/fetch_real_data.py --ticker GOOGL --years 5
python scripts/fetch_real_data.py --ticker TSLA --years 5
python scripts/fetch_real_data.py --ticker INFY.NS --years 5
python scripts/fetch_real_data.py --ticker TCS.NS --years 5
```

### 4. Ingest documents into vector store

```bash
python scripts/ingest_all.py
# Expected: ~842 vectors indexed across 23 files
```

### 5. Load CSVs into SQLite database

```bash
python scripts/load_sql_db.py
# Creates data/processed/fin.db with 9 tables
```

### 6. Run the API and UI

```bash
# Terminal 1 — API
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — UI
python ui/app.py
```

Open **http://127.0.0.1:7860** in your browser.

---

## Project Structure

```
fin-analyzer/
├── config.py                    ← Pydantic settings (reads .env)
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
│   └── main.py                  ← FastAPI: /ingest /ask /eda /regression /analyze
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
│   ├── raw/                     ← Source files (CSVs, TXTs, PDFs)
│   ├── processed/               ← SQLite database (fin.db)
│   └── embeddings/              ← FAISS index files
│
├── reports/
│   └── output/                  ← RAGAS eval results, generated PDFs
│
├── .env.example                 ← Environment template
├── requirements.txt             ← Python dependencies
├── Dockerfile                   ← Container definition
└── docker-compose.yml           ← Multi-service orchestration
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

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API status + vector count |
| `POST` | `/ingest` | Upload and index a document |
| `POST` | `/ask` | Natural language Q&A via agent |
| `POST` | `/eda` | EDA on uploaded CSV/Excel |
| `POST` | `/regression` | Linear regression on uploaded CSV |
| `POST` | `/analyze` | Full pipeline: EDA + regression + ingest |

Interactive docs available at **http://127.0.0.1:8000/docs**

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
python -m pytest tests/ -v
# Expected: 29 passed
```

---

## Docker

```bash
cp .env.example .env
# Add your OpenAI API key to .env

docker compose up --build
# UI → http://localhost:7860
# API → http://localhost:8000/docs
```

---

## Known Limitations

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
| UI | Gradio 6 |
| Visualization | Plotly, Matplotlib |
| Testing | Pytest (29 tests) |
| Containers | Docker, Docker Compose |

---

## Author

**Bhavika Sharma** — Data Science & LLM Developer  
[LinkedIn](http://www.linkedin.com/in/bhavika-sharma-ml) · [GitHub](https://github.com/bhavika67/) · [Hugging Face](https://huggingface.co/Bhavika67)
