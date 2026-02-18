"""
Cleans raw product text for downstream embedding and classification:
    - Removes HTML tags, URLs, emails, SKUs, product references, long numbers.
    - Normalizes whitespace and lowercases.
    - Pre-compiled Regex for performance on large batches.
"""

import html
import re
from typing import Any

import pandas as pd

# Pre-compiled regex patterns
REGEX_HTML = re.compile(r"<[^>]+>")
REGEX_URL = re.compile(r"http[s]?://\S+|www\.\S+")
REGEX_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
REGEX_PRODUCT_REF = re.compile(
    r"\b(ref|réf|reference|référence|fabricant|model|modèle|sku|asin|ean|isbn|gtin)"
    r"[\s\.:_-]*[a-z0-9][a-z0-9\-_.]{2,}\b"
)
REGEX_CODE = re.compile(r"[a-zA-Z]{2,}-\d+(?:-[a-zA-Z]{2,})?")
REGEX_LONG_NUM = re.compile(r"\b\d{6,}\b")
REGEX_SPECIAL_CHARS = re.compile(r"[^\w\sàâäéèêëïîôùûüÿçœæ]")
REGEX_SPACES = re.compile(r"\s+")


def clean_text(text: str | Any = None) -> str:
    """
    Clean a single text string by removing noise
    and normalizing whitespace.
    Returns empty string for None/NaN inputs.
    """
    if text is None or pd.isna(text):
        return ""

    text = html.unescape(str(text)).lower()
    text = REGEX_HTML.sub(" ", text)
    text = REGEX_URL.sub(" ", text)
    text = REGEX_EMAIL.sub(" ", text)
    text = REGEX_PRODUCT_REF.sub(" ", text)
    text = REGEX_CODE.sub(" ", text)
    text = REGEX_LONG_NUM.sub(" ", text)
    text = REGEX_SPECIAL_CHARS.sub(" ", text)
    text = REGEX_SPACES.sub(" ", text).strip()

    return text


def input_text_train(
    df: pd.DataFrame,
    col_des: str = "product_designation",
    col_desc: str = "product_description",
) -> pd.DataFrame:
    """
    Clean designation and description columns,
    then build a unified 'input_text' field for training.
    Returns a copy of the DataFrame with the new column.
    """
    df = df.copy()
    designation_clean = df[col_des].map(clean_text)
    description_clean = df[col_desc].map(clean_text)

    df["input_text"] = (designation_clean + " " + description_clean).str.strip()

    return df


def input_text_infer(designation: str, description: str | None = None) -> str:
    """
    Build a unified text field for single-record inference.
    Mirrors the training preprocessing for consistency.
    """
    return (clean_text(designation) + " " + clean_text(description)).strip()
