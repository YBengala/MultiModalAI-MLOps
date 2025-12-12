from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image


def test_encode_dataframe_batches(train_encoder):
    """Verifies correct batch processing and final output reassembly matching input size."""
    # Setup DataFrame (3 images)
    df = pd.DataFrame({"path": ["img1.jpg", "img2.jpg", "img3.jpg"]})

    # Mock image opening to avoid disk I/O
    with patch("PIL.Image.open") as mock_open:
        # Return a solid blue image (224x224)
        mock_open.return_value = Image.new("RGB", (224, 224), color="blue")

        embeddings = train_encoder.image_train_encodings(df, path_column="path")

        # Check output type
        assert isinstance(embeddings, np.ndarray)
        # Check total number of vectors (should match row count)
        assert embeddings.shape[0] == 3
        # Check embedding dimension (e.g., 512 for ResNet18)
        assert embeddings.shape[1] == train_encoder.get_embedding_dim()

        # Verify DataLoader behavior (Image.open should be called once per image)
        assert mock_open.call_count == 3


def test_encode_empty_dataframe(train_encoder):
    """Ensures empty DataFrames return a properly shaped empty array (0, D) without crashing."""
    df = pd.DataFrame({"path": []})

    embeddings = train_encoder.image_train_encodings(df, path_column="path")

    assert isinstance(embeddings, np.ndarray)
    # Should return an empty array with shape (0, Dim)
    assert embeddings.shape == (0, train_encoder.get_embedding_dim())


def test_encode_with_corrupted_image(train_encoder):
    """Tests robustness: ensures fallback usage on I/O errors without pipeline failure."""
    df = pd.DataFrame({"path": ["good.jpg", "bad.jpg"]})

    with patch("PIL.Image.open") as mock_open:
        # Define side effects: First call works, second raises Error
        good_img = Image.new("RGB", (224, 224), color="blue")
        mock_open.side_effect = [good_img, IOError("Corrupted file")]

        embeddings = train_encoder.image_train_encodings(df, path_column="path")

        # We must still have 2 embeddings (one real, one fallback)
        assert embeddings.shape[0] == 2
        # The pipeline did not crash
        assert not np.isnan(embeddings).any()
