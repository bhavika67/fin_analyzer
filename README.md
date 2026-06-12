# Financial Document Analyzer

An agentic RAG application for analyzing financial documents — annual reports,
quarterly P&L statements, stock price histories, and company profiles. Ingest
PDFs, Word docs, CSVs, and text files into a vector store, ask natural language
questions via a LangGraph ReAct agent, and run EDA, regression, and anomaly
detection on tabular financial data.

## Features

- **Document ingestion** — parse PDF, DOCX, CSV/XLSX, and TXT files into
  chunks and index them in a FAISS vector store.
- **Agentic Q&A** — a LangGraph ReAct agent with tools for searching and
  summarizing financial documents (revenue, margins, stock prices, company
  profiles, annual reports).
- **Exploratory Data Analysis (EDA)** — summary statistics, missing values,
  correlations, trend analysis, and auto-generated insights for any uploaded
  CSV/Excel file.
- **Regression analysis** — fit a linear regression against a target column
  and get R², MAE, RMSE, coefficients, and a plain-English interpretation.
- **Anomaly detection** — IQR and Z-score based outlier detection across
  numeric columns.
- **Gradio UI** — tabs for Ingest, Ask, EDA, and Regression, each with
  Plotly visualizations (trend charts, correlation charts, anomaly charts,
  coefficient charts, R² gauge).
- **RAG evaluation** — score the agent's answers with RAGAS
  (faithfulness, answer relevancy, context precision).

## Project Structure

```
fin_analyzer/
├── agent/
│   └── agent.py            # LangGraph ReAct agent + tools
├── api/
│   └── main.py             # FastAPI app (ingest, ask, eda, regression)
├── ingestion/
│   ├── parser.py           # PDF/DOCX/CSV/XLSX/TXT → ParsedDocument
│   ├── chunker.py           # ParsedDocument → Chunks
│   └── pipeline.py         # parse + chunk in one step
├── vectorstore/
│   ├── embedder.py          # OpenAI embeddings wrapper
│   └── store.py             # FAISS index with save/load
├── eda/
│   ├── analyzer.py          # EDAAnalyzer — stats, correlations, trends, insights
│   ├── regression.py        # RegressionAnalyzer — linear regression
│   └── anomaly.py            # AnomalyDetector — IQR / Z-score outliers
├── evaluation/
│   └── evaluator.py          # RAGEvaluator — RAGAS scoring
├── ui/
│   ├── app.py                # Gradio frontend
│   └── charts.py             # Plotly chart builders (dark theme)
├── scripts/
│   ├── generate_sample_data.py  # synthetic sample CSVs + annual report
│   ├── fetch_real_data.py       # pull real ticker data via yfinance
│   ├── ingest_all.py            # bulk-ingest everything in data/raw/
│   ├── ask.py                   # CLI Q&A smoke test
│   └── evaluate_rag.py          # run RAGAS evaluation on sample questions
├── tests/
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_eda.py
│   └── test_vectorstore.py
├── config.py                 # pydantic Settings (.env driven)
├── requirements.txt
└── .env                       # not committed — see Setup below
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

VECTOR_STORE=faiss
FAISS_INDEX_PATH=data/embeddings/faiss_index

DATABASE_URL=sqlite:///data/processed/fin.db
LOG_LEVEL=INFO
```

### 4. Generate or fetch sample data

Synthetic data (quarterly P&L, segment revenue, cost/headcount, narrative
annual report):

```bash
python scripts/generate_sample_data.py
```

Real data for a specific ticker (prices, quarterly financials, company
profile) via Yahoo Finance:

```bash
python scripts/fetch_real_data.py --ticker MSFT --years 5
```

### 5. Ingest documents into the vector store

```bash
python scripts/ingest_all.py
```

This parses and chunks everything in `data/raw/` and saves the FAISS index
to `data/embeddings/faiss_index/`.

## Running the App

### Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### Start the UI

```bash
python ui/app.py
```

By default the UI talks to `http://127.0.0.1:8000` — override with the
`API_BASE` environment variable if the API runs elsewhere.

## API Endpoints

| Method | Path          | Description                                  |
|--------|---------------|-----------------------------------------------|
| GET    | `/health`     | Health check + total vectors indexed          |
| POST   | `/ingest`     | Upload a document (PDF/DOCX/CSV/XLSX/TXT)     |
| POST   | `/ask`        | Ask a natural language question               |
| POST   | `/eda`        | Run EDA on an uploaded CSV/Excel file         |
| POST   | `/regression` | Run linear regression on an uploaded CSV/Excel |

## CLI Tools

```bash
# Interactive Q&A smoke test (runs a fixed set of sample questions)
python scripts/ask.py

# RAGAS evaluation against a ground-truth Q&A set
python scripts/evaluate_rag.py
```

`evaluate_rag.py` writes results to `reports/output/eval_results.json`.

## Testing

```bash
pytest
```

Tests cover document parsing/chunking (`test_ingestion.py`), EDA/regression/
anomaly detection (`test_eda.py`), and the vector store using a mocked
embedder so no OpenAI API calls are made (`test_vectorstore.py`).

## Troubleshooting

### RAGAS evaluation (`scripts/evaluate_rag.py`) — dependency & API issues

This project was tested against `ragas==0.4.3` (the `vibrantlabsai` fork),
`langchain==1.3.8`, `langchain-core==1.4.6`, `langchain-openai==1.3.0`, and a
current `langchain-community`. Getting this combination working required two
fixes:

**1. `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`**

This `ragas` build unconditionally imports `ChatVertexAI` from
`langchain_community.chat_models.vertexai`, even if you never use Vertex AI.
Current `langchain-community` releases don't ship that module. Create a stub
so the import succeeds (the class is never instantiated unless you configure
a Vertex AI LLM):

Find your environment's `site-packages/langchain_community/chat_models/`
directory and add `vertexai.py`:

```python
class ChatVertexAI:
    pass
```

On Windows (PowerShell), e.g.:

```powershell
$dir = "<path-to-site-packages>\langchain_community\chat_models"
New-Item -ItemType File -Path "$dir\vertexai.py" -Force -Value "class ChatVertexAI:`n    pass`n"
```

⚠️ This stub lives inside your installed package and will be wiped out if
you reinstall/upgrade `langchain-community` — recreate it after any such
upgrade.

**2. `ragas` >=0.2 schema change (`evaluation/evaluator.py`)**

Newer `ragas` versions changed the evaluation dataset schema and require
explicit LLM/embeddings wrappers:

- Dataset columns are now `user_input` / `response` / `retrieved_contexts` /
  `reference` instead of the old `question` / `answer` / `contexts` /
  `ground_truth`, and use `EvaluationDataset.from_list(...)` instead of a HF
  `Dataset`.
- `evaluate()` needs `llm=LangchainLLMWrapper(ChatOpenAI(...))` and
  `embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings(...))` passed
  explicitly — without these you'll see
  `AttributeError: 'OpenAIEmbeddings' object has no attribute 'embed_query'`.
- The faithfulness check on long answers can hit
  `finish_reason='length'` (`The output is incomplete due to a max_tokens
  length limit`) with the default token cap — set a higher `max_tokens`
  (e.g. `4096`) on the evaluator LLM.

`evaluation/evaluator.py` already implements all of the above.

### Agent gives "please specify the company" for document-level questions

Questions like *"What were the key risks in the annual report?"* or *"What
was the total revenue for FY 2023?"* refer to the single narrative annual
report (`annual_report_fy2023.txt`), not a specific ticker. If the agent asks
you to specify a company for these, it's a retrieval/prompt issue — consider
adding the document's company/ticker context to its metadata, or adjusting
the system prompt in `agent/agent.py` to recognize document-level (non-ticker)
queries. This currently shows up as `answer_relevancy: 0.0` for those samples
in the RAGAS report.

### Low `context_precision` on comparison/ranking questions

Questions like *"Which company had the highest net income?"* or *"TCS vs
Infosys revenue"* can score `context_precision: 0.0` even when the agent's
final answer is correct — the agent may use `summarize_financial_topic`
across several searches internally, but `evaluate_rag.py` only captures the
top-3 chunks from a single `store.search()` call as the RAGAS "contexts" for
that sample, which may not match what the agent actually used.

## Tech Stack

- **FastAPI** — REST API
- **LangChain / LangGraph** — agent orchestration (ReAct pattern)
- **OpenAI** — LLM (`gpt-4o`) and embeddings (`text-embedding-3-small`)
- **FAISS** — vector similarity search
- **pandas / scikit-learn / scipy** — EDA, regression, anomaly detection
- **Gradio + Plotly** — interactive UI and charts
- **RAGAS** — RAG quality evaluation
- **pytest** — testing
