# ui/app.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

import gradio as gr
import httpx
from ui.charts import (
    empty_chart, trend_chart, correlation_chart,
    anomaly_chart, coefficient_chart, r2_gauge
)

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TIMEOUT  = 120


# ── API helpers ───────────────────────────────────────────────────────────────

def ingest_file(file) -> str:
    if file is None:
        return "No file selected."
    path = Path(file)
    with open(path, "rb") as f:
        resp = httpx.post(
            f"{API_BASE}/ingest",
            files={"file": (path.name, f)},
            timeout=TIMEOUT,
        )
    if resp.status_code == 200:
        d = resp.json()
        return (f"**{d['filename']}** ingested successfully.\n"
                f"- Chunks indexed: **{d['chunks_indexed']}**\n"
                f"- Total vectors in store: **{d['total_vectors']}**")
    return f"Error {resp.status_code}: {resp.text}"


def ask_question(question: str, history: list) -> tuple:
    if not question.strip():
        return history, ""
    resp = httpx.post(
        f"{API_BASE}/ask",
        json={"question": question},
        timeout=TIMEOUT,
    )
    if resp.status_code == 200:
        d      = resp.json()
        answer = d.get("answer") or f"Error: {d.get('error')}"
    else:
        answer = f"API error {resp.status_code}: {resp.text}"
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, ""


def run_eda(file, target_col: str):
    """Returns: summary text, trend chart, correlation chart, anomaly chart."""
    empty = empty_chart("Run analysis to see chart")

    if file is None:
        return "No file selected.", empty, empty, empty

    path = Path(file)
    with open(path, "rb") as f:
        params = {}
        if target_col.strip():
            params["target_column"] = target_col.strip()
        resp = httpx.post(
            f"{API_BASE}/eda",
            files={"file": (path.name, f)},
            params=params,
            timeout=TIMEOUT,
        )
    if resp.status_code != 200:
        return f"Error {resp.status_code}: {resp.text}", empty, empty, empty

    d = resp.json()

    # ── Text summary ──────────────────────────────────────────
    lines = [
        f"## EDA Results — {path.name}",
        f"**Shape:** {d['shape']['rows']} rows x {d['shape']['cols']} cols\n",
        "### Insights",
    ]
    for ins in d.get("insights", []):
        lines.append(f"- {ins}")
    if d.get("missing_values"):
        lines.append("\n### Missing Values")
        for col, info in d["missing_values"].items():
            lines.append(f"- **{col}**: {info['count']} missing ({info['pct']}%)")
    summary = "\n".join(lines)

    # ── Build charts ──────────────────────────────────────────
    t = trend_chart(d.get("trends", {}))
    c = correlation_chart(d.get("correlations", {}))
    a = anomaly_chart(d.get("anomalies", []))

    print(f"Trend chart traces:  {len(t.data)}")
    print(f"Corr chart traces:   {len(c.data)}")
    print(f"Anomaly chart traces:{len(a.data)}")

    return summary, t, c, a


def run_regression(file, target_col: str):
    """Returns: summary text, coefficient chart, R2 gauge."""
    empty = empty_chart("Run regression to see chart")

    if file is None:
        return "No file selected.", empty, empty
    if not target_col.strip():
        return "Please enter a target column name.", empty, empty

    path = Path(file)
    with open(path, "rb") as f:
        resp = httpx.post(
            f"{API_BASE}/regression",
            files={"file": (path.name, f)},
            params={"target_column": target_col.strip()},
            timeout=TIMEOUT,
        )
    if resp.status_code != 200:
        return f"Error {resp.status_code}: {resp.text}", empty, empty

    d = resp.json()

    # ── Text summary ──────────────────────────────────────────
    lines = [
        f"## Regression Results — target: `{d['target']}`\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| R2     | **{d['r2']}** |",
        f"| MAE    | {d['mae']} |",
        f"| RMSE   | {d['rmse']} |",
        f"\n**{d['interpretation']}**",
    ]
    summary = "\n".join(lines)

    return (
        summary,
        coefficient_chart(d.get("coefficients", {})),
        r2_gauge(d.get("r2", 0)),
    )


# ── UI Layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Financial Document Analyzer") as demo:

    gr.Markdown(
        "# Financial Document Analyzer\n"
        "Ingest reports · Ask questions · Run EDA & Regression"
    )

    with gr.Tabs():

        # ── Ingest ────────────────────────────────────────────
        with gr.Tab("Ingest"):
            gr.Markdown("Upload a PDF, DOCX, CSV, or Excel file to add it to the knowledge base.")
            ingest_input  = gr.File(label="Select Document",
                                    file_types=[".pdf", ".docx", ".csv", ".xlsx", ".txt"])
            ingest_btn    = gr.Button("Ingest Document", variant="primary")
            ingest_output = gr.Markdown()
            ingest_btn.click(ingest_file,
                             inputs=ingest_input,
                             outputs=ingest_output)

        # ── Ask ───────────────────────────────────────────────
        with gr.Tab("Ask"):
            gr.Markdown("Ask natural language questions about your ingested documents.")
            chatbot = gr.Chatbot(height=440, label="Chat")
            q_input = gr.Textbox(
                placeholder="e.g. Compare INFY and TCS net margins",
                label="Your question",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")
            ask_btn.click(ask_question,
                          inputs=[q_input, chatbot],
                          outputs=[chatbot, q_input])
            q_input.submit(ask_question,
                           inputs=[q_input, chatbot],
                           outputs=[chatbot, q_input])

        # ── EDA ───────────────────────────────────────────────
        with gr.Tab("EDA"):
            gr.Markdown("Upload a CSV or Excel file for Exploratory Data Analysis.")
            with gr.Row():
                eda_file   = gr.File(label="Upload CSV / Excel",
                                     file_types=[".csv", ".xlsx"])
                eda_target = gr.Textbox(
                    placeholder="e.g. net_income (optional)",
                    label="Target column",
                )
            eda_btn    = gr.Button("Run Analysis", variant="primary")
            eda_output = gr.Markdown(label="Summary")
            with gr.Row():
                trend_plot = gr.Plot(label="Trends",       min_width=300)
                corr_plot  = gr.Plot(label="Correlations", min_width=300)
            anom_plot  = gr.Plot(label="Anomalies Detected")
            eda_btn.click(
                fn=run_eda,
                inputs=[eda_file, eda_target],
                outputs=[eda_output, trend_plot, corr_plot, anom_plot],
            )

        # ── Regression ────────────────────────────────────────
        with gr.Tab("Regression"):
            gr.Markdown("Run linear regression on a numeric CSV dataset.")
            with gr.Row():
                reg_file   = gr.File(label="Upload CSV",
                                     file_types=[".csv", ".xlsx"])
                reg_target = gr.Textbox(
                    placeholder="e.g. net_profit",
                    label="Target column (required)",
                )
            reg_btn    = gr.Button("Run Regression", variant="primary")
            reg_output = gr.Markdown(label="Summary")
            with gr.Row():
                coef_plot  = gr.Plot(label="Feature Coefficients", min_width=300)
                gauge_plot = gr.Plot(label="R2 Score",              min_width=300)
            reg_btn.click(
                fn=run_regression,
                inputs=[reg_file, reg_target],
                outputs=[reg_output, coef_plot, gauge_plot],
            )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )