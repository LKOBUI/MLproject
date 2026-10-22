"""Gradio inference UI. It loads a trained model; it does not train on startup."""
from __future__ import annotations
import os
from pathlib import Path
import gradio as gr
import torch
from app.model import load_model, predict_tensor
from app.utils import preprocess_image

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device(os.getenv("MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "saved_model.pth"))
CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", ROOT / "models" / "model_config.json"))
MODEL = None
LOAD_ERROR = None
try:
    MODEL = load_model(MODEL_PATH, CONFIG_PATH, DEVICE)
except Exception as exc:  # keep UI able to explain missing model file
    LOAD_ERROR = str(exc)


def classify(image):
    if MODEL is None:
        raise gr.Error(f"Model could not be loaded: {LOAD_ERROR}")
    tensor = preprocess_image(image)
    result = predict_tensor(MODEL, tensor, DEVICE)
    return str(result["digit"]), result["confidence"], result["probabilities"]


with gr.Blocks(title="Digit Mist — PyTorch") as demo:
    gr.Markdown("# Digit Mist — PyTorch\nDraw or upload a handwritten digit (0–9).")
    if LOAD_ERROR:
        gr.Markdown(
            "**Model not ready:** Train/export a model first with "
            "`python -m scripts.train`. Details: " + LOAD_ERROR
        )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Draw or upload a digit")
        with gr.Column():
            digit_output = gr.Textbox(label="Predicted digit")
            confidence_output = gr.Number(label="Predicted-class confidence")
            probabilities_output = gr.Label(label="Class probabilities", num_top_classes=10)
    predict_button = gr.Button("Predict", variant="primary")
    predict_button.click(classify, inputs=image_input,
                         outputs=[digit_output, confidence_output, probabilities_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
