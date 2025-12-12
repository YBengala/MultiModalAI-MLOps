import pandas as pd

from multimodal_ai.features.text_cleaner import (
    clean_text,
    input_text_infer,
    input_text_train,
)


def test_clean_text_basic():
    """Verifies basic HTML cleaning, entity decoding, and whitespace normalization."""
    raw = "  <H1>Bonjour</H1>   MONDE&nbsp;!  "
    expected = "bonjour monde !"
    assert clean_text(raw) == expected


def test_remove_sensitive_info():
    """Ensures privacy-sensitive patterns (emails, URLs) are stripped from the text."""
    raw = "Contactez moi@test.com ou visitez https://site.com"
    clean = clean_text(raw)
    assert "moi@test.com" not in clean
    assert "https://site.com" not in clean
    assert "contactez" in clean


def test_remove_product_references():
    """Validates the removal of specific identifiers (Product Refs, ISBNs, IDs)."""
    # Explicit reference
    assert clean_text("Produit ref:A123-BC") == "produit"
    # ISBN
    assert clean_text("Livre isbn 978-3-16-148410-0") == "livre"
    # Long number (internal ID)
    assert clean_text("Commande 123456789") == "commande"


def test_input_text_train():
    """Checks DataFrame column concatenation and null handling for training preparation."""
    df = pd.DataFrame(
        {"designation": ["Prod A", None], "description": ["Desc A", "Desc B"]}
    )

    df_res = input_text_train(df, col_des="designation", col_desc="description")

    assert "input_text" in df_res.columns
    assert df_res.iloc[0]["input_text"] == "prod a desc a"
    assert df_res.iloc[1]["input_text"] == "desc b"


def test_input_text_infer():
    """Verifies text preparation logic (cleaning + merging) for inference inputs."""
    res = input_text_infer(designation="SUPER <br> Produit", description="Ref:123")
    assert res == "super produit"
