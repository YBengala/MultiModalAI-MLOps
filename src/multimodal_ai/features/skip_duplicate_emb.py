"""
Skip Duplicate Embeddings – Rakuten MLOps Pipeline
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

EMBEDDINGS_FILE = Path("/data/embeddings/embeddings.parquet")


def has_new_products(processed_csv: Path) -> bool:
    batch_df = pd.read_csv(processed_csv, sep=";")
    batch_ids = set(batch_df["productid"].astype(int))

    if not EMBEDDINGS_FILE.exists():
        logger.info("No embeddings file yet → embed all %d products", len(batch_ids))
        return True

    try:
        existing_table = pq.read_table(EMBEDDINGS_FILE, columns=["product_id"])
        existing_ids = set(existing_table.column("product_id").to_pylist())
    except Exception as e:
        logger.warning("Could not read embeddings file: %s → embed all", e)
        return True

    if not batch_ids:
        logger.info("No products in batch → SKIP")
        return False

    new_ids = batch_ids - existing_ids
    already_embedded = batch_ids & existing_ids
    logger.info(
        "Batch: %d new products, %d to re-embed (content may have changed)",
        len(new_ids),
        len(already_embedded),
    )
    return True
