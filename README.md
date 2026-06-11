# Financial Document Analyzer

An agentic RAG (Retrieval-Augmented Generation) system for ingesting financial documents, asking natural language questions, and running EDA and regression analysis — powered by LangGraph, FAISS, and OpenAI.

---

## Features

- **Document Ingestion** — Parse and index PDFs, DOCX, CSV, Excel, and TXT files into a FAISS vector store
- **Agentic Q&A** — ReAct agent (LangGraph) answers questions by retrieving and synthesizing content across documents
- **Exploratory Data Analysis** — Summary stats, correlations, trend detection, structural break detection, and insights
- **Anomaly Detection** — IQR and Z-score based outlier detection on numeric data
- **Linear Regression** — Feature importance, R², MAE, RMSE, and natural language interpretation
- **REST API** — FastAPI backend with `/ingest`, `/ask`, `/eda`, and `/regression` endpoints
- **Gradio UI** — Browser-based interface with tabs for each feature

---

## Project Structure

```
fin-analyzer/
├── agent/
│   └── agent.py              # LangGraph ReAct agent with search tools
├── api/
│   └── main.py               # FastAPI app — all HTTP endpoints
├── eda/
│   ├── analyzer.py           # EDAAnalyzer — stats, trends, correlations
│   ├── anomaly.py            # AnomalyDetector — IQR & Z-score
│   └── regression.py         # RegressionAnalyzer — LinearRegression wrapper
├── ingestion/
│   ├── parser.py             # DocumentParser — PDF, DOCX, CSV, Excel, TXT
│   ├── chunker.py            # TextChunker — RecursiveCharacterTextSplitter
│   └── pipeline.py           # IngestionPipeline — parse → chunk
├── vectorstore/
│   ├── embedder.py           # OpenAI text-embedding-3-small
│   └── store.py              # FAISS vector store with save/load
├── scripts/
│   ├── ask.py                # CLI Q&A against the vector store
│   ├── fetch_real_data.py    # Fetch live data from Yahoo Finance
│   ├── generate_sample_data.py # Generate synthetic CSV/TXT fixtures
│   └── ingest_all.py         # Batch ingest all files in data/raw/
├── tests/
│   ├── conftest.py           # Shared pytest fixtures
│   ├── test_eda.py           # EDA, anomaly, regression tests
│   ├── test_ingestion.py     # Parser and chunker tests
│   └── test_vectorstore.py   # Vector store tests (mocked embedder)
├── ui/
│   └── app.py                # Gradio frontend
├── config.py                 # Pydantic settings (reads .env)
├── requirements.txt
└── .env                      # API keys and config (not committed)
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-username/fin-analyzer.git
cd fin-analyzer
python -m pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
FAISS_INDEX_PATH=data/embeddings/faiss_index
```

### 3. Generate or fetch data

```bash
# Option A — synthetic sample data (no API key needed)
python scripts/generate_sample_data.py

# Option B — real data from Yahoo Finance
python scripts/fetch_real_data.py --ticker AAPL --years 5
python scripts/fetch_real_data.py --ticker MSFT --years 5
```

### 4. Ingest documents into the vector store

```bash
python scripts/ingest_all.py
```

### 5. Start the API server

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Launch the Gradio UI

```bash
python ui/app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## API Reference

All endpoints are available at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + vector count |
| `POST` | `/ingest` | Upload and index a document |
| `POST` | `/ask` | Ask a natural language question |
| `POST` | `/eda` | Run EDA on a CSV/Excel file |
| `POST` | `/regression` | Run regression on a CSV/Excel file |

### Example: Ask a question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the net profit margin in FY2023?"}'
```

### Example: Ingest a document

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@data/raw/annual_report_fy2023.txt"
```

### Example: Run EDA

```bash
curl -X POST "http://127.0.0.1:8000/eda?target_column=net_profit" \
  -F "file=@data/raw/quarterly_pl.csv"
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests mock the OpenAI embedder — no API key is required to run them.

```
tests/test_eda.py          # 12 tests — EDA, anomaly detection, regression
tests/test_ingestion.py    # 6 tests  — parser, chunker
tests/test_vectorstore.py  # 9 tests  — add, search, save, load
```

---

## Configuration

All settings are managed via `config.py` using Pydantic and read from `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Chat model for the agent |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `FAISS_INDEX_PATH` | `data/embeddings/faiss_index` | Where to persist the FAISS index |
| `DATABASE_URL` | `sqlite:///data/processed/fin.db` | SQLite DB path (reserved) |
| `LOG_LEVEL` | `INFO` | Loguru log level |

---

## Supported File Types

| Extension | Parser |
|-----------|--------|
| `.pdf` | pypdf |
| `.docx` | python-docx |
| `.csv` | pandas |
| `.xlsx` / `.xls` | pandas + openpyxl |
| `.txt` | built-in |

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Agent | LangGraph, LangChain, OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (facebook/faiss-cpu) |
| API | FastAPI + Uvicorn |
| UI | Gradio |
| EDA / ML | pandas, numpy, scikit-learn, scipy |
| Data Fetch | yfinance |
| Config | pydantic-settings |
| Logging | loguru |
| Testing | pytest |
