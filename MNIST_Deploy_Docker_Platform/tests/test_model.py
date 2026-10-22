import json
import torch
from app.model import PytorchTensorDigitMist


def test_model_output_shape():
    model = PytorchTensorDigitMist(hidden=300)
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 28, 28))
    assert tuple(output.shape) == (1, 10)


def test_probabilities_sum_to_one():
    model = PytorchTensorDigitMist(hidden=300).eval()
    with torch.inference_mode():
        probs = torch.softmax(model(torch.zeros(1, 1, 28, 28)), dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-6)
