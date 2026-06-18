# ui/app.py
import sys
import re
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
        print("RAW ANSWER:", repr(answer))
    else:
        answer = f"API error {resp.status_code}: {resp.text}"

    history.append({"role": "user", "content": question})

    # ── Check if the answer contains a generated chart ─────────
    chart_match = re.search(r"CHART_GENERATED:(.+?\.png)\|(.+)", answer)
    if chart_match:
        chart_path  = chart_match.group(1).strip()
        chart_title = chart_match.group(2).strip().split("\n")[0]
        chart_title = re.sub(r"[\)\]]+$", "", chart_title)  # strip trailing ) or ]

        # Normalize path for Gradio
        chart_path = str(Path(chart_path).resolve())

        # Any text before the marker, with markdown image syntax stripped
        before = answer[:chart_match.start()]
        before = re.sub(r"!\[[^\]]*\]\($", "", before).strip()

        if before:
            history.append({"role": "assistant", "content": before})

        if Path(chart_path).exists():
            history.append({"role": "assistant", "content": {"path": chart_path}})
            history.append({"role": "assistant", "content": chart_title})
        else:
            history.append({"role": "assistant",
                            "content": f"Chart was generated but file not found at: {chart_path}"})
    else:
        history.append({"role": "assistant", "content": answer})

    return history, ""


def get_table_choices() -> list:
    """Fetch the list of tables currently loaded in the financial database."""
    try:
        resp = httpx.get(f"{API_BASE}/tables", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("tables", [])
    except Exception:
        pass
    return []


def _call_analysis_endpoint(endpoint: str, source: str, file_path, table_name: str,
                             params: dict) -> httpx.Response:
    """POST to /eda or /regression with either an uploaded file or a table_name."""
    if source == "Database Table":
        return httpx.post(f"{API_BASE}/{endpoint}",
                           params={**params, "table_name": table_name},
                           timeout=TIMEOUT)
    with open(file_path, "rb") as f:
        return httpx.post(f"{API_BASE}/{endpoint}",
                           files={"file": (Path(file_path).name, f)},
                           params=params, timeout=TIMEOUT)


def run_analysis(source: str, file, table_name: str, target_col: str):
    """Combined EDA + optional Regression on an uploaded file or a database table.
    Returns: summary markdown, trend, correlation, anomaly, coefficient, r2 gauge."""
    empty = empty_chart("Run analysis to see chart")
    target_col = (target_col or "").strip()

    if source == "Database Table":
        if not table_name:
            return "Please select a table.", empty, empty, empty, empty, empty
        label = table_name
    else:
        if file is None:
            return "No file selected.", empty, empty, empty, empty, empty
        label = Path(file).name

    # ── EDA (always runs) ────────────────────────────────────────
    eda_params = {"target_column": target_col} if target_col else {}
    resp = _call_analysis_endpoint("eda", source, file, table_name, eda_params)
    if resp.status_code != 200:
        return f"Error {resp.status_code}: {resp.text}", empty, empty, empty, empty, empty

    d = resp.json()
    lines = [
        f"## EDA Results — {label}",
        f"**Shape:** {d['shape']['rows']} rows x {d['shape']['cols']} cols\n",
        "### Insights",
    ]
    for ins in d.get("insights", []):
        lines.append(f"- {ins}")
    if d.get("missing_values"):
        lines.append("\n### Missing Values")
        for col, info in d["missing_values"].items():
            lines.append(f"- **{col}**: {info['count']} missing ({info['pct']}%)")

    trend_fig = trend_chart(d.get("trends", {}))
    corr_fig  = correlation_chart(d.get("correlations", {}))
    anom_fig  = anomaly_chart(d.get("anomalies", []))
    coef_fig  = empty_chart("Add a target column above to run regression")
    gauge_fig = empty_chart("Add a target column above to run regression")

    # ── Regression (only if a target column was given) ────────────
    if target_col:
        reg_resp = _call_analysis_endpoint("regression", source, file, table_name,
                                            {"target_column": target_col})
        if reg_resp.status_code == 200:
            rd = reg_resp.json()
            lines += [
                f"\n### Regression — target: `{rd['target']}`",
                "| Metric | Value |",
                "|--------|-------|",
                f"| R² | **{rd['r2']}** |",
                f"| MAE | {rd['mae']} |",
                f"| RMSE | {rd['rmse']} |",
                f"\n**{rd['interpretation']}**",
            ]
            coef_fig  = coefficient_chart(rd.get("coefficients", {}))
            gauge_fig = r2_gauge(rd.get("r2", 0))
        else:
            lines.append(f"\n*Regression failed: {reg_resp.text}*")

    return "\n".join(lines), trend_fig, corr_fig, anom_fig, coef_fig, gauge_fig


# ── UI Layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Financial Document Analyzer") as demo:

    gr.Markdown(
        "# Financial Document Analyzer\n"
        "Build a knowledge base and ask questions · Analyze any dataset"
    )

    with gr.Tabs():

        # ── Knowledge Base & Ask ─────────────────────────────────
        with gr.Tab("Knowledge Base & Ask"):
            gr.Markdown(
                "Add a PDF, DOCX, CSV, or Excel file to the knowledge base, then ask "
                "questions about it. You can also ask for charts, e.g. *'Show me a "
                "line chart of revenue from quarterly_pl'*."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    ingest_input  = gr.File(label="Add Document",
                                            file_types=[".pdf", ".docx", ".csv", ".xlsx", ".txt"])
                    ingest_btn    = gr.Button("Add to Knowledge Base", variant="secondary")
                    ingest_output = gr.Markdown()
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=480, label="Chat")
                    q_input = gr.Textbox(
                        placeholder="e.g. Compare INFY and TCS net margins, or 'plot revenue trend from quarterly_pl'",
                        label="Your question",
                        lines=2,
                    )
                    ask_btn = gr.Button("Ask", variant="primary")

            ingest_btn.click(ingest_file, inputs=ingest_input, outputs=ingest_output)
            ask_btn.click(ask_question, inputs=[q_input, chatbot], outputs=[chatbot, q_input])
            q_input.submit(ask_question, inputs=[q_input, chatbot], outputs=[chatbot, q_input])

        # ── Analyze (EDA + Regression, file or database table) ──
        with gr.Tab("Analyze"):
            gr.Markdown(
                "Run EDA (and optional regression) on an uploaded CSV/Excel file, "
                "or directly on a table already loaded into the financial database."
            )
            source_radio = gr.Radio(
                ["Upload File", "Database Table"],
                value="Upload File",
                label="Data source",
            )
            with gr.Row():
                analyze_file = gr.File(label="Upload CSV / Excel",
                                       file_types=[".csv", ".xlsx"], visible=True)
                with gr.Column(visible=False) as table_col:
                    analyze_table      = gr.Dropdown(label="Database table",
                                                      choices=get_table_choices())
                    refresh_tables_btn = gr.Button("↻ Refresh tables", size="sm")
                analyze_target = gr.Textbox(
                    placeholder="e.g. net_profit (leave blank to skip regression)",
                    label="Target column (optional)",
                )
            analyze_btn    = gr.Button("Run Analysis", variant="primary")
            analyze_output = gr.Markdown(label="Summary")
            with gr.Row():
                trend_plot = gr.Plot(label="Trends",       min_width=260)
                corr_plot  = gr.Plot(label="Correlations", min_width=260)
                anom_plot  = gr.Plot(label="Anomalies",    min_width=260)
            with gr.Row():
                coef_plot  = gr.Plot(label="Feature Coefficients", min_width=300)
                gauge_plot = gr.Plot(label="R² Score",              min_width=300)

            def toggle_source(choice):
                is_file = (choice == "Upload File")
                return gr.update(visible=is_file), gr.update(visible=not is_file)

            source_radio.change(
                toggle_source,
                inputs=source_radio,
                outputs=[analyze_file, table_col],
            )

            refresh_tables_btn.click(
                lambda: gr.update(choices=get_table_choices()),
                outputs=analyze_table,
            )

            analyze_btn.click(
                fn=run_analysis,
                inputs=[source_radio, analyze_file, analyze_table, analyze_target],
                outputs=[analyze_output, trend_plot, corr_plot, anom_plot, coef_plot, gauge_plot],
            )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )