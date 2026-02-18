"""
Skip Duplicate Batch – Rakuten MLOps Pipeline
"""

import logging
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("/data/processed")
EMBEDDINGS_FILE = Path("/data/embeddings/embeddings.parquet")


def is_batch_already_processed(run_id: str) -> bool:
    # Processed CSV exists
    processed_csv = PROCESSED_DIR / f"{run_id}.csv"
    if not processed_csv.exists():
        logger.info("Batch %s: no processed CSV → new batch", run_id)
        return False

    # Embeddings contain run_id
    if EMBEDDINGS_FILE.exists():
        try:
            table = pq.read_table(EMBEDDINGS_FILE, columns=["run_id"])
            existing_run_ids = set(table.column("run_id").to_pylist())
            if run_id in existing_run_ids:
                logger.warning("Batch %s: already fully processed → SKIP", run_id)
                return True
        except Exception as e:
            logger.warning("Could not read embeddings file: %s", e)

    logger.info(
        "Batch %s: processed CSV exists but embeddings incomplete → reprocess",
        run_id,
    )
    return False
