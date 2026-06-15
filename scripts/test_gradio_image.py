# scripts/test_gradio_image.py
import gradio as gr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
charts = list((ROOT / "ui" / "charts").glob("*.png"))
chart_path = str(charts[0].resolve()) if charts else None
print("Testing with:", chart_path)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        value=[
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": {"path": chart_path}},
        ],
        height=400,
    )

demo.launch(server_name="127.0.0.1", server_port=7861)