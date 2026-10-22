"""Evaluate exported weights on the MNIST test set and save metrics/plot."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.model import load_model

ROOT = Path(__file__).resolve().parents[1]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST(ROOT / "data", train=False, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model = load_model(ROOT / "models/saved_model.pth", ROOT / "models/model_config.json", device)
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().tolist()
            y_pred.extend(pred); y_true.extend(y.tolist())
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred)),
               "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)}
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=list(range(10)))
    disp.plot(cmap=None, colorbar=False); plt.tight_layout()
    plt.savefig(ROOT / "artifacts/confusion_matrix.png", dpi=140); plt.close()
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Saved metrics and confusion matrix in {ROOT / 'artifacts'}")

if __name__ == "__main__": main()
