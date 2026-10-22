"""Validate the exported state_dict/config pair. Training writes these files automatically."""
from pathlib import Path
import json
import torch
from app.model import load_model
ROOT = Path(__file__).resolve().parents[1]

def main():
    config_path = ROOT / "models/model_config.json"
    weights_path = ROOT / "models/saved_model.pth"
    if not config_path.exists() or not weights_path.exists():
        raise SystemExit("No exported model found. Run `python -m scripts.train` first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = load_model(weights_path, config_path, "cpu")
    with torch.inference_mode():
        output = model(torch.zeros((1, 1, 28, 28)))
    if tuple(output.shape) != (1, 10):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")
    print(f"Validated {weights_path} ({config.get('model_name', 'unnamed')}); output shape={tuple(output.shape)}")

if __name__ == "__main__": main()
