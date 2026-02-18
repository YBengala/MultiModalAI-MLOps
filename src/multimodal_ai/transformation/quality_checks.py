"""
Validates processed data before passing to embedding stage.
Raises exceptions if thresholds are not met.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds
MIN_IMAGE_MATCH_RATE = 0.90  # 90% of products must have a valid image
MIN_ROW_COUNT = 100  # Minimum rows expected per batch


def check_volume(df: pd.DataFrame, run_id: str, min_rows: int = MIN_ROW_COUNT) -> None:
    """Verify the batch has a minimum number of rows."""
    if len(df) < min_rows:
        raise ValueError(
            f"Quality gate failed: batch {run_id} has {len(df)} rows, "
            f"minimum is {min_rows}"
        )
    logger.info("Volume check passed: %d rows (min: %d)", len(df), min_rows)


def check_image_match_rate(
    df: pd.DataFrame,
    run_id: str,
    min_rate: float = MIN_IMAGE_MATCH_RATE,
) -> None:
    """Verify that enough products have matching images."""
    total = len(df)
    matched = df["image_exists"].sum()
    rate = matched / total if total > 0 else 0

    if rate < min_rate:
        raise ValueError(
            f"Quality gate failed: batch {run_id} image match rate "
            f"{rate:.1%} < {min_rate:.1%} ({matched}/{total})"
        )
    logger.info(
        "Image match check passed: %d / %d (%.1f%%)", matched, total, rate * 100
    )


def check_null_designation(df: pd.DataFrame, run_id: str) -> None:
    """Verify no empty designations after cleaning."""
    empty = (df["designation"] == "").sum()
    if empty > 0:
        logger.warning(
            "Batch %s has %d products with empty designation after cleaning",
            run_id,
            empty,
        )


def run_quality_checks(df: pd.DataFrame, run_id: str) -> None:
    """Run all quality check"""
    logger.info("Running quality checks for batch %s", run_id)
    check_volume(df, run_id)
    check_image_match_rate(df, run_id)
    check_null_designation(df, run_id)
    logger.info("All quality checks passed for batch %s", run_id)
