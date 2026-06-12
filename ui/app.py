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


def run_eda(file, target_col: str) -> str:
    if file is None:
        return "No file selected."
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
        return f"Error {resp.status_code}: {resp.text}"
    d     = resp.json()
    lines = [f"## EDA Results — {path.name}",
             f"**Shape:** {d['shape']['rows']} rows x {d['shape']['cols']} cols\n"]
    lines.append("### Insights")
    for ins in d.get("insights", []):
        lines.append(f"- {ins}")
    if d.get("anomalies"):
        lines.append("\n### Anomalies Detected")
        for a in d["anomalies"]:
            lines.append(f"- {a['summary']}")
    if d.get("correlations"):
        lines.append("\n### Strong Correlations")
        for pair, val in list(d["correlations"].items())[:6]:
            lines.append(f"- {pair}: **{val:.3f}**")
    if d.get("trends"):
        lines.append("\n### Trends")
        for col, t in list(d["trends"].items())[:3]:
            lines.append(f"- **{col}**: {t['direction']} "
                         f"(avg change: {t['avg_pct_change']}%)")
    return "\n".join(lines)


def run_regression(file, target_col: str) -> str:
    if file is None:
        return "No file selected."
    if not target_col.strip():
        return "Please enter a target column name."
    path = Path(file)
    with open(path, "rb") as f:
        resp = httpx.post(
            f"{API_BASE}/regression",
            files={"file": (path.name, f)},
            params={"target_column": target_col.strip()},
            timeout=TIMEOUT,
        )
    if resp.status_code != 200:
        return f"Error {resp.status_code}: {resp.text}"
    d     = resp.json()
    lines = [
        f"## Regression Results — target: `{d['target']}`\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| R2     | **{d['r2']}** |",
        f"| MAE    | {d['mae']} |",
        f"| RMSE   | {d['rmse']} |",
        f"\n**{d['interpretation']}**",
        "\n### Feature Coefficients",
    ]
    for feat, coef in sorted(d["coefficients"].items(),
                             key=lambda x: abs(x[1]), reverse=True):
        bar = "+" * min(int(abs(coef) * 5), 20)
        lines.append(f"- `{feat}`: {coef:+.4f}  {bar}")
    return "\n".join(lines)


# ── UI Layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Financial Document Analyzer") as demo:

    gr.Markdown(
        "# Financial Document Analyzer\n"
        "Ingest reports · Ask questions · Run EDA & Regression"
    )

    with gr.Tabs():

        with gr.Tab("Ingest"):
            gr.Markdown("Upload a PDF, DOCX, CSV, or Excel file to add it to the knowledge base.")
            ingest_input  = gr.File(label="Select Document",
                                    file_types=[".pdf", ".docx", ".csv", ".xlsx", ".txt"])
            ingest_btn    = gr.Button("Ingest Document", variant="primary")
            ingest_output = gr.Markdown()
            ingest_btn.click(ingest_file,
                             inputs=ingest_input,
                             outputs=ingest_output)

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

        with gr.Tab("EDA"):
            gr.Markdown("Upload a CSV or Excel file for instant Exploratory Data Analysis.")
            with gr.Row():
                eda_file   = gr.File(label="Upload CSV / Excel",
                                     file_types=[".csv", ".xlsx"])
                eda_target = gr.Textbox(
                    placeholder="e.g. net_income (optional)",
                    label="Target column",
                )
            eda_btn    = gr.Button("Run EDA", variant="primary")
            eda_output = gr.Markdown()
            eda_btn.click(run_eda,
                          inputs=[eda_file, eda_target],
                          outputs=eda_output)

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
            reg_output = gr.Markdown()
            reg_btn.click(run_regression,
                          inputs=[reg_file, reg_target],
                          outputs=reg_output)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )