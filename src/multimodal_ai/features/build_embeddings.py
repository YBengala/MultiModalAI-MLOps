"""
Reads processed CSV, generates text + image embeddings incrementally,
appends to /data/embeddings/embeddings.parquet via /data/tmp/
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multimodal_ai.features.base_text_embedder import BaseTextEmbedder
from multimodal_ai.features.image_encoder_train import ImageEncoderTrain
from multimodal_ai.transformation.text_cleaner import input_text_train

PROCESSED_DIR = Path("/data/processed")
EMBEDDINGS_DIR = Path("/data/embeddings")
TMP_DIR = Path("/data/tmp")
RAW_IMAGES_DIR = Path("/data/raw/images")

EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.parquet"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


# Load processed CSV
def load_processed_csv(run_id: str) -> pd.DataFrame:
    """Load the processed CSV for a given batch."""
    csv_path = PROCESSED_DIR / f"{run_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")

    processed_data = pd.read_csv(csv_path, sep=";")
    logger.info("Loaded %d rows from %s", len(processed_data), csv_path)
    return processed_data


# Filter products with valid images
def filter_valid_products(processed_data: pd.DataFrame) -> pd.DataFrame:
    """Keep only products with existing images for embedding."""
    valid_data = processed_data[processed_data["image_exists"].astype(bool)].copy()
    logger.info(
        "Valid products for embedding: %d / %d",
        len(valid_data),
        len(processed_data),
    )
    return pd.DataFrame(valid_data)


# Build input text for embedding
def prepare_text_input(valid_data: pd.DataFrame) -> pd.DataFrame:
    """Concatenate designation + description into input_text."""
    valid_data = input_text_train(
        valid_data,
        col_des="designation",
        col_desc="description",
    )
    logger.info("Text input prepared: %d rows", len(valid_data))
    return valid_data


# Generate text embeddings
def generate_text_embeddings(valid_data: pd.DataFrame) -> np.ndarray:
    """Encode text using Solon sentence-transformer."""
    text_encoder = BaseTextEmbedder()
    texts = valid_data["input_text"].fillna("").tolist()
    text_embeddings = text_encoder.encode_text(texts)
    logger.info("Text embeddings generated: %s", text_embeddings.shape)
    return text_embeddings


# Generate image embeddings
def generate_image_embeddings(valid_data: pd.DataFrame, run_id: str) -> np.ndarray:
    """Encode images using EfficientViT via timm."""
    # Build local image paths from raw images directory
    images_dir = RAW_IMAGES_DIR / run_id
    valid_data = valid_data.copy()
    valid_data["local_image_path"] = valid_data.apply(
        lambda row: str(
            images_dir
            / f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg"
        ),
        axis=1,
    )

    image_encoder = ImageEncoderTrain()
    image_embeddings = image_encoder.image_train_encodings(
        valid_data, path_column="local_image_path"
    )
    logger.info("Image embeddings generated: %s", image_embeddings.shape)
    return image_embeddings


# Build batch Parquet in /data/tmp/
def build_batch_parquet(
    valid_data: pd.DataFrame,
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    run_id: str,
) -> Path:
    """Create a temporary Parquet file for the current batch."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_DIR / f"batch_embeddings_{run_id}.parquet"

    batch_records = pd.DataFrame(
        {
            "product_id": valid_data["productid"].astype(int).values,
            "embedding_text": [row.tolist() for row in text_embeddings],
            "embedding_image": [row.tolist() for row in image_embeddings],
            "prdtypecode": valid_data["prdtypecode"].astype(int).values,
            "run_id": run_id,
        }
    )

    batch_records.to_parquet(tmp_path, index=False)
    logger.info("Batch Parquet saved: %s (%d rows)", tmp_path, len(batch_records))
    return tmp_path


# Append batch to main embeddings Parquet (with deduplication on product update)
def append_to_embeddings(batch_path: Path) -> int:
    """
    Append the batch Parquet to the main embeddings file.
    If a product_id already exists, keep the latest version (new batch wins).
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    batch_table = pq.read_table(batch_path)
    batch_df = batch_table.to_pandas()
    batch_rows = len(batch_df)

    if EMBEDDINGS_FILE.exists():
        existing_df = pq.read_table(EMBEDDINGS_FILE).to_pandas()
        nb_before = len(existing_df)

        # Remove existing products that are in the new batch (update case)
        updated_ids = set(batch_df["product_id"])
        existing_df = existing_df[~existing_df["product_id"].isin(updated_ids)]
        nb_replaced = nb_before - len(existing_df)

        combined_df = pd.concat([existing_df, batch_df], ignore_index=True)

        if nb_replaced > 0:
            logger.info(
                "Product update: %d existing embeddings replaced by new batch",
                nb_replaced,
            )
    else:
        combined_df = batch_df
        nb_replaced = 0

    combined_table = pa.Table.from_pandas(combined_df, preserve_index=False)
    pq.write_table(combined_table, EMBEDDINGS_FILE)

    logger.info(
        "Embeddings appended: +%d new, %d replaced (total: %d)",
        batch_rows - nb_replaced,
        nb_replaced,
        len(combined_df),
    )
    return batch_rows
