"""Train the notebook-compatible MLP on MNIST, save weights, and log to MLflow."""
from __future__ import annotations
import json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import mlflow
from mlflow.tracking import MlflowClient
from app.model import PytorchTensorDigitMist, PytorchTensorDigitMistLoRA

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
DB_PATH = ROOT / "mlflow.db"
REGISTERED_MODEL_NAME = "Digit Mist Pytorch Cuda"


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DB_PATH.as_posix()}")


def main():
    seed = int(os.getenv("SEED", "42"))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    hidden = int(os.getenv("HIDDEN", "300"))
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    epochs = int(os.getenv("EPOCHS", "5"))
    lr = float(os.getenv("LEARNING_RATE", "0.001"))
    kernel_init = os.getenv("KERNEL_INIT", "GlorotUniform")
    lora_uses = os.getenv("LORA_USES", "false").strip().lower() in {"1", "true", "yes"}
    lora_rank = int(os.getenv("LORA_RANK", "8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_root = ROOT / "data"
    train_full = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_ds, val_ds = random_split(train_full, [55000, 5000], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    if lora_uses:
        model = PytorchTensorDigitMistLoRA(hidden=hidden, kernel_init=kernel_init, r=lora_rank).to(device)
    else:
        model = PytorchTensorDigitMist(hidden=hidden, kernel_init=kernel_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", REGISTERED_MODEL_NAME))
    MODEL_DIR.mkdir(parents=True, exist_ok=True); ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "epoch_seconds": []}
    best_val_loss = float("inf")
    best_state = None
    with mlflow.start_run(run_name="digit_mist_training") as run:
        mlflow.log_params({"architecture": "PytorchTensorDigitMist", "hidden": hidden,
                           "use_bias": True, "kernel_init": kernel_init,
                           "batch_size": batch_size, "epochs_requested": epochs,
                           "learning_rate": lr, "seed": seed, "device": str(device),
                           "lora_uses": lora_uses, "lora_rank": lora_rank if lora_uses else 0,
                           "normalization_mean": 0.1307, "normalization_std": 0.3081})
        for epoch in range(1, epochs + 1):
            start = time.time(); model.train(); total_loss = 0.0; seen = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images); loss = criterion(logits, labels)
                loss.backward(); optimizer.step()
                total_loss += loss.item() * labels.size(0); seen += labels.size(0)
            train_loss = total_loss / max(seen, 1)
            model.eval(); val_total = 0.0; val_seen = 0; correct = 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images); loss = criterion(logits, labels)
                    val_total += loss.item() * labels.size(0); val_seen += labels.size(0)
                    correct += (logits.argmax(1) == labels).sum().item()
            val_loss = val_total / max(val_seen, 1); val_acc = correct / max(val_seen, 1)
            elapsed = time.time() - start
            history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc); history["epoch_seconds"].append(elapsed)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                "val_accuracy": val_acc, "epoch_seconds": elapsed}, step=epoch)
            print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        # Evaluate on the held-out test split.
        model.eval(); test_correct = 0; test_seen = 0; test_loss_total = 0.0
        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images); loss = criterion(logits, labels)
                test_loss_total += loss.item() * labels.size(0); test_seen += labels.size(0)
                test_correct += (logits.argmax(1) == labels).sum().item()
        test_loss = test_loss_total / max(test_seen, 1); test_acc = test_correct / max(test_seen, 1)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})
        torch.save(model.cpu().state_dict(), MODEL_DIR / "saved_model.pth")
        config = {"model_name": REGISTERED_MODEL_NAME, "architecture": "PytorchTensorDigitMist",
                  "hidden": hidden, "use_bias": True, "kernel_init": kernel_init,
                  "lora_uses": lora_uses, "lora_rank": lora_rank if lora_uses else 0,
                  "input_shape": [1, 1, 28, 28], "num_classes": 10,
                  "normalization_mean": 0.1307, "normalization_std": 0.3081,
                  "mlflow_run_id": run.info.run_id}
        (MODEL_DIR / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (ARTIFACT_DIR / "metrics.json").write_text(json.dumps({**history, "test_loss": test_loss,
             "test_accuracy": test_acc, "best_val_loss": best_val_loss}, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(MODEL_DIR / "model_config.json"), artifact_path="model_export")
        mlflow.log_artifact(str(ARTIFACT_DIR / "metrics.json"), artifact_path="evaluation")
        mlflow.log_artifact(str(MODEL_DIR / "saved_model.pth"), artifact_path="model_export")
        mlflow.set_tag("exported_model_path", "models/saved_model.pth")
        print(f"Test loss: {test_loss:.4f}; test accuracy: {test_acc:.4f}")
        print(f"Saved weights: {MODEL_DIR / 'saved_model.pth'}")
        print(f"MLflow run ID: {run.info.run_id}")
        # Register the logged PyTorch model artifact. State-dict export remains the deployment file.
        try:
            mlflow.pytorch.log_model(model, artifact_path="pytorch_model", serialization_format="pickle",
                                     input_example=torch.zeros((1, 1, 28, 28), dtype=torch.float32))
            model_uri = f"runs:/{run.info.run_id}/pytorch_model"
            mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
            print(f"Registered model: {REGISTERED_MODEL_NAME}, version {mv.version}")
        except Exception as exc:
            # Training and export still succeed if registry setup is unavailable.
            print(f"WARNING: MLflow model registration did not complete: {exc}")

if __name__ == "__main__":
    main()
