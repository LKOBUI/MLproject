from PIL import Image
from app.utils import preprocess_image


def test_preprocessing_shape():
    tensor = preprocess_image(Image.new("RGB", (50, 50), color="white"))
    assert tuple(tensor.shape) == (1, 1, 28, 28)


def test_missing_image_is_rejected():
    try:
        preprocess_image(None)
    except ValueError:
        return
    raise AssertionError("None image should raise ValueError")
