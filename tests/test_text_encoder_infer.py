import numpy as np


def test_encode_full_product(infer_text_encoder):
    """Tests encoding with both designation and description."""
    designation = "Chaussures Nike"
    description = "Modèle <br> Air Max taille 42."

    embedding = infer_text_encoder.encode_text_infer(designation, description)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] == infer_text_encoder.get_embedding_dim()
    assert not np.isnan(embedding).any()


def test_encode_designation_only(infer_text_encoder):
    """Tests encoding when description is None."""
    designation = "Produit simple"

    embedding = infer_text_encoder.encode_text_infer(designation, description=None)

    assert isinstance(embedding, np.ndarray)
    assert not np.isnan(embedding).any()
