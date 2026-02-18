"""
Logs pipeline data metrics to MLflow after each batch run.
Tracks: ingestion volumes, transformation quality, embedding stats,
        encoder models used, embedding dimensions,
        and cumulative embedding metrics.
"""

import logging
import os
from pathlib import Path

import mlflow

from multimodal_ai.config.settings import settings

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", settings.MLFLOW_EXPERIMENT_NAME)
EMBEDDINGS_FILE = Path("/data/embeddings/embeddings.parquet")


def _get_cumulative_stats() -> dict:
    """Read embeddings Parquet and return cumulative stats."""
    if not EMBEDDINGS_FILE.exists():
        return {}

    try:
        from collections import Counter

        import pyarrow.parquet as pq

        table = pq.read_table(EMBEDDINGS_FILE, columns=["prdtypecode"])
        codes = table.column("prdtypecode").to_pylist()

        if not codes:
            return {}

        counts = Counter(codes)
        return {
            "nb_classes": len(counts),
            "total_products": sum(counts.values()),
            "imbalance_ratio": round(
                max(counts.values()) / max(min(counts.values()), 1), 2
            ),
        }
    except Exception as e:
        logger.warning("Could not read embeddings file: %s", e)
        return {}


def log_pipeline_run(metadata: dict) -> str:
    """
    Log a full pipeline data run to MLflow.

    Logs encoder models, embedding dimensions, ingestion/transformation
    volumes, and cumulative embedding stats.

    Args:
        metadata: dict accumulated through the DAG containing:
            - run_id, nb_images, nb_rows_inserted
            - nb_images_found, nb_images_uploaded, nb_processed_inserted
            - nb_embeddings, nb_appended
            - quality_passed
            - dt_ingested, dt_transformed, dt_processed, dt_embeddings

    Returns:
        MLflow run_id string.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_id = metadata.get("run_id", "unknown")
    cumul_stats = _get_cumulative_stats()

    with mlflow.start_run(run_name=f"pipeline_{run_id}") as run:
        # Parameters: Batch + Models + Dimensions
        mlflow.log_params(
            {
                "run_id": run_id,
                "nb_images_ingested": metadata.get("nb_images", 0),
                "nb_embeddings_total": metadata.get("nb_appended", 0),
                "text_model": settings.TEXT_MODEL_NAME,
                "image_model": settings.IMAGE_MODEL_NAME,
                "embedding_text_dim": settings.TEXT_EMBEDDING_DIM,
                "embedding_image_dim": settings.IMAGE_EMBEDDING_DIM,
                "fusion_dim": settings.fusion_dim,
                "device": settings.DEFAULT_DEVICE,
            }
        )

        # Metrics: Ingestion
        if "nb_rows_inserted" in metadata:
            mlflow.log_metric(
                "ingestion.nb_rows_inserted", metadata["nb_rows_inserted"]
            )

        # Metrics: Transformation
        if "nb_images_found" in metadata:
            mlflow.log_metric("transform.nb_images_found", metadata["nb_images_found"])
        if "nb_images_uploaded" in metadata:
            mlflow.log_metric(
                "transform.nb_images_uploaded", metadata["nb_images_uploaded"]
            )
        if "nb_processed_inserted" in metadata:
            mlflow.log_metric(
                "transform.nb_processed_inserted", metadata["nb_processed_inserted"]
            )

        # Metrics: Embeddings (batch)
        if "nb_embeddings" in metadata:
            mlflow.log_metric("embeddings.nb_generated", metadata["nb_embeddings"])
        if "nb_appended" in metadata:
            mlflow.log_metric("embeddings.nb_appended", metadata["nb_appended"])

        # Metrics: Cumulative (all batches)
        if cumul_stats:
            mlflow.log_metric(
                "embeddings.cumulated_products", cumul_stats["total_products"]
            )
            mlflow.log_metric("embeddings.cumulated_classes", cumul_stats["nb_classes"])
            mlflow.log_metric(
                "embeddings.class_imbalance_ratio", cumul_stats["imbalance_ratio"]
            )

        # Tags
        mlflow.set_tags(
            {
                "pipeline": "rakuten_ingestion",
                "quality_passed": str(metadata.get("quality_passed", False)),
            }
        )

        # Timestamps
        for key in ("dt_ingested", "dt_transformed", "dt_processed", "dt_embeddings"):
            if key in metadata:
                mlflow.log_param(key, metadata[key])

        mlflow_run_id = run.info.run_id
        logger.info(
            "MLflow run logged: %s (experiment: %s)", mlflow_run_id, EXPERIMENT_NAME
        )
        return mlflow_run_id
