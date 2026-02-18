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

    new_ids = batch_ids - existing_ids
    if new_ids:
        logger.info(
            "Found %d new products to embed (out of %d)", len(new_ids), len(batch_ids)
        )
        return True

    logger.info("All %d products already embedded → SKIP", len(batch_ids))
    return False
