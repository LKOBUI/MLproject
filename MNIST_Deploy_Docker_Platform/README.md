# Digit Mist — PyTorch + Gradio + MLflow deployment

This project turns the notebook's MNIST MLP into a small Python application with separate training, evaluation, export validation, and inference code. The network architecture follows the notebook's `PytorchTensorDigitMist`: flatten 28×28 grayscale input, two ReLU hidden layers, and 10 output logits. Optional LoRA architecture support is included when `LORA_USES=true`.

## 1. Environment setup

Use Python 3.12 and install dependencies compatible with your platform (the development machine in the notebook is Ubuntu ARM64):

```bash
conda activate mlx
python -m pip install -r docker/requirements.txt
```

For CUDA, install the PyTorch/torchvision builds matching your installed NVIDIA driver and CUDA runtime from the official PyTorch selector rather than assuming the default pip build is CUDA-enabled.

## 2. Train and create deployment files

From the project root:

```bash
python -m scripts.train
```

The script downloads MNIST if needed, trains the model, evaluates on the test set, logs parameters/metrics/artifacts to MLflow, writes `models/saved_model.pth` and `models/model_config.json`, and attempts to register `Digit Mist Pytorch Cuda` in the registry.

Defaults are a non-LoRA MLP with hidden=300 and 5 epochs to keep the first run practical. You can configure it through environment variables, e.g.:

```bash
HIDDEN=300 EPOCHS=10 BATCH_SIZE=128 LORA_USES=false python -m scripts.train
```

For LoRA, use `LORA_USES=true LORA_RANK=8`; `loralib` must be installed. Do not change architecture flags after training unless you train and export a matching model again.

MLflow defaults to SQLite at `sqlite:///.../deploymodels/mlflow.db`. Override with `MLFLOW_TRACKING_URI` when using a tracking server. Keep the SQLite database and artifact storage persistent; do not commit the database to source control.

## 3. Validate and evaluate

```bash
python -m scripts.export_model
python -m scripts.evaluate
python -m pytest tests/ -v
```

Evaluation creates `artifacts/metrics.json` and `artifacts/confusion_matrix.png`.

## 4. Run Gradio locally

```bash
python -m app.interface
# or: ./run.sh
```

Open `http://localhost:7860`. The inference app loads the saved weights on startup and does not retrain when users make predictions.

## 5. Docker

The Docker image expects the trained `models/saved_model.pth` and `models/model_config.json` to exist first. Build from the project root:

```bash
docker build --platform linux/arm64 -f docker/Dockerfile -t digit-mist-app:latest .
docker run --rm --name digit-mist -p 7860:7860 digit-mist-app:latest
```

Use `linux/amd64` instead if deploying to an x86-64 server. Check that PyTorch/torchvision wheels are available for the target architecture. This inference container does not need the MLflow database mounted because it loads the exported local model. Keep MLflow tracking/registry in the training environment or a separately configured tracking server.

## Project structure

- `app/`: model, preprocessing, and Gradio inference UI.
- `scripts/train.py`: MNIST training, metrics, MLflow logging and registration, local export.
- `scripts/evaluate.py`: held-out test evaluation and confusion matrix.
- `scripts/export_model.py`: validates weights/config match and expected output shape.
- `models/`: generated state dict and architecture/preprocessing config.
- `configs/`: sample runtime and logging settings.
- `tests/`: basic model and preprocessing tests.

## Important notes

- `saved_model.pth` is a `state_dict`; it does not contain the Python model architecture. The architecture is reconstructed from `app/model.py` and `models/model_config.json`.
- The notebook's Gradio UI also supports training controls. This deployment UI intentionally focuses on inference; training is a separate CLI workflow.
- The first training run downloads MNIST, so internet access is required at training time. Inference itself can run offline once the model files and image-processing dependencies are installed.
- Validate accuracy and preprocessing on real drawn digits before exposing the app publicly.
