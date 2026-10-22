#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f models/saved_model.pth ]]; then
  echo "No trained model found. Training MNIST model first..."
  python -m scripts.train
fi
python -m scripts.export_model
exec python -m app.interface
