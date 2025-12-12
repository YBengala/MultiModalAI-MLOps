import numpy as np
import pandas as pd


def test_encode_dataframe_columns(train_text_encoder):
    """Tests embedding extraction from a DataFrame with separate columns."""
    df = pd.DataFrame(
        {
            "titre": ["Produit A", "Produit B"],
            "desc": ["Super description", "Autre description"],
        }
    )
    embeddings = train_text_encoder.text_train_encodings(
        df, col_designation="titre", col_description="desc"
    )

    assert isinstance(embeddings, np.ndarray)
    # 2 lignes -> 2 vecteurs
    assert embeddings.shape[0] == len(df)
    assert embeddings.shape[1] == train_text_encoder.get_embedding_dim()
    assert not np.isnan(embeddings).any()


def test_encode_empty_dataframe(train_text_encoder):
    """Verifies empty input."""
    df = pd.DataFrame({"titre": [], "desc": []})
    embeddings = train_text_encoder.text_train_encodings(df, "titre", "desc")
    assert embeddings.shape == (0, train_text_encoder.get_embedding_dim())
