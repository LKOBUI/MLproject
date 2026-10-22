"""PyTorch model architecture and inference helpers matching the notebook MLP."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F

_KERNEL_INITS = {"GlorotNormal", "GlorotUniform", "HeNormal", "HeUniform"}
_HIDDEN_CHOICES = {1000, 750, 500, 300, 150, 75, 25}


def _apply_kernel_init(tensor: torch.Tensor, init_name: str) -> None:
    if init_name == "GlorotNormal":
        nn.init.xavier_normal_(tensor)
    elif init_name == "GlorotUniform":
        nn.init.xavier_uniform_(tensor)
    elif init_name == "HeNormal":
        nn.init.kaiming_normal_(tensor, nonlinearity="relu")
    elif init_name == "HeUniform":
        nn.init.kaiming_uniform_(tensor, nonlinearity="relu")
    else:
        raise ValueError(f"Unknown kernel initializer: {init_name}")


class PytorchTensorDigitMist(nn.Module):
    """The notebook's MLP: 784 -> hidden -> hidden//2 -> 10 logits."""
    def __init__(self, hidden: int = 300, use_bias: bool = True,
                 kernel_init: str = "GlorotUniform"):
        super().__init__()
        if hidden not in _HIDDEN_CHOICES:
            raise ValueError(f"hidden must be one of {sorted(_HIDDEN_CHOICES)}")
        if kernel_init not in _KERNEL_INITS:
            raise ValueError(f"kernel_init must be one of {sorted(_KERNEL_INITS)}")
        self.hidden = int(hidden)
        self.use_bias = bool(use_bias)
        self.kernel_init = kernel_init
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden, bias=use_bias)
        self.fc2 = nn.Linear(hidden, hidden // 2, bias=use_bias)
        self.fc3 = nn.Linear(hidden // 2, 10, bias=use_bias)
        for layer in (self.fc1, self.fc2, self.fc3):
            _apply_kernel_init(layer.weight, kernel_init)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_model(model_path: str | Path = "models/saved_model.pth",
               config_path: str | Path = "models/model_config.json",
               device: str | torch.device = "cpu") -> PytorchTensorDigitMist:
    """Recreate the configured architecture and safely load its state_dict."""
    model_path, config_path = Path(model_path), Path(config_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained weights not found at {model_path}. Run `python -m scripts.train` first."
        )
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    model_kwargs = {
        "hidden": int(config.get("hidden", 300)),
        "use_bias": bool(config.get("use_bias", True)),
        "kernel_init": str(config.get("kernel_init", "GlorotUniform")),
    }
    if bool(config.get("lora_uses", False)):
        model = PytorchTensorDigitMistLoRA(
            **model_kwargs, r=int(config.get("lora_rank", 8))
        )
    else:
        model = PytorchTensorDigitMist(**model_kwargs)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_tensor(model: nn.Module, image_tensor: torch.Tensor,
                   device: str | torch.device = "cpu") -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        logits = model(image_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    digit = int(torch.argmax(probs).item())
    return {
        "digit": digit,
        "confidence": float(probs[digit].item()),
        "probabilities": {str(i): float(probs[i].item()) for i in range(10)},
    }

# Optional LoRA architecture matching the notebook's PytorchTensorDigitMistLoRA.
try:
    import loralib as lora
except ImportError:  # non-LoRA deployment does not require loralib at import time
    lora = None

class PytorchTensorDigitMistLoRA(PytorchTensorDigitMist):
    def __init__(self, hidden: int = 300, use_bias: bool = True,
                 kernel_init: str = "GlorotUniform", r: int = 8):
        if lora is None:
            raise ImportError("LoRA model requested but loralib is not installed. Install loralib.")
        super().__init__(hidden=hidden, use_bias=use_bias, kernel_init=kernel_init)
        self.fc1 = lora.Linear(28 * 28, hidden, r=r, bias=use_bias)
        self.fc2 = lora.Linear(hidden, hidden // 2, r=r, bias=use_bias)
        self.fc3 = lora.Linear(hidden // 2, 10, r=r, bias=use_bias)
