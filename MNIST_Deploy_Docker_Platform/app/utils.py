"""Input preprocessing and small utilities for MNIST inference."""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

# Must match the notebook's MNIST normalization.
_PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def preprocess_image(image) -> torch.Tensor:
    """Convert a PIL/numpy Gradio image into (1, 1, 28, 28) normalized tensor."""
    if image is None:
        raise ValueError("Please draw or upload a digit first.")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image or NumPy image array.")
    # Gradio's drawing canvas is often white background/black ink, as in MNIST.
    image = image.convert("RGB")
    tensor = _PREPROCESS(image)
    return tensor.unsqueeze(0)
