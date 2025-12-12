from __future__ import annotations

import pandas as pd
from PIL import Image

from multimodal_ai.config.settings import settings
from multimodal_ai.features.image_dataset import ImageDataset


def test_dataset_initialization():
    """Verifies dataset length and path extraction from the input DataFrame."""
    df = pd.DataFrame({"path": ["img1.jpg", "img2.jpg"]})
    ds = ImageDataset(df, path_column="path")
    assert len(ds) == 2
    assert ds.image_paths == ["img1.jpg", "img2.jpg"]


def test_load_valid_image(tmp_path):
    """Ensures valid images are correctly loaded and decoded using a temporary file."""
    # Create a temporary dummy image
    d = tmp_path / "images"
    d.mkdir()
    image_file = d / "test_valid.png"
    Image.new("RGB", (10, 10)).save(image_file)

    # Test loading
    df = pd.DataFrame({"filepath": [str(image_file)]})
    ds = ImageDataset(df, path_column="filepath")
    loaded_img = ds[0]

    assert isinstance(loaded_img, Image.Image)
    assert loaded_img.size[0] > 0


def test_load_missing_image_fallback():
    """Validates the white-placeholder fallback mechanism for missing or broken paths."""
    df = pd.DataFrame({"filepath": ["/path/to/nonexistent/img.jpg"]})
    ds = ImageDataset(df, path_column="filepath")

    fallback_img = ds[0]

    assert isinstance(fallback_img, Image.Image)
    expected_size = settings.IMAGE_SIZE
    assert fallback_img.size == (expected_size, expected_size)
    center_pixel = fallback_img.getpixel((expected_size // 2, expected_size // 2))
    assert center_pixel == (255, 255, 255)


def test_transform_application(tmp_path):
    """Verifies that the custom transform function is correctly applied to the image."""
    # Create a temporary dummy image
    d = tmp_path / "images"
    d.mkdir()
    image_file = d / "test_transform.png"
    Image.new("RGB", (10, 10)).save(image_file)

    # Define a mock transform
    def dummy_transform(img):
        return "TRANSFORMED"

    df = pd.DataFrame({"filepath": [str(image_file)]})
    ds = ImageDataset(df, path_column="filepath", transform=dummy_transform)

    output = ds[0]
    assert output == "TRANSFORMED"
