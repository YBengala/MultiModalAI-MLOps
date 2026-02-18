"""
Reads raw data from products_raw, cleans text, validates images,
uploads images to MinIO (skips existing), updates category_mapping.json,
saves clean CSV to /data/processed/ and loads products_processed table.
"""

import json
import logging
import os
from pathlib import Path

import boto3
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from multimodal_ai.transformation.text_cleaner import clean_text

# Configuration
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("RAKUTEN_DB_HOST", "rakuten-postgres"),
    "dbname": os.getenv("RAKUTEN_DB_NAME"),
    "user": os.getenv("RAKUTEN_DB_USER"),
    "password": os.getenv("RAKUTEN_DB_PASSWORD"),
}

MINIO_CONFIG = {
    "endpoint_url": os.getenv("MINIO_ENDPOINT_URL", "http://rakuten-minio:9000"),
    "aws_access_key_id": os.getenv("MINIO_ROOT_USER"),
    "aws_secret_access_key": os.getenv("MINIO_ROOT_PASSWORD"),
}

MINIO_BUCKET = "rakuten-datalake"
MINIO_IMAGES_PREFIX = "images/"

RAW_DIR = Path("/data/raw")
RAW_IMAGES_DIR = Path("/data/raw/images")
PROCESSED_DIR = Path("/data/processed")
CATEGORY_MAPPING_PATH = PROCESSED_DIR / "category_mapping.json"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


# S3 Client (MinIO)
def get_s3_client():
    """Create a boto3 S3 client configured for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=MINIO_CONFIG["endpoint_url"],
        aws_access_key_id=MINIO_CONFIG["aws_access_key_id"],
        aws_secret_access_key=MINIO_CONFIG["aws_secret_access_key"],
    )


# Load raw CSV
def load_raw_csv(run_id: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the raw CSV from /data/raw/{run_id}.csv."""
    csv_path = raw_dir / f"{run_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.lower().str.strip()
    df["batch_id"] = run_id
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


# Remove duplicates
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on productid, keep first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=["productid"], keep="first")
    removed = before - len(df)
    if removed > 0:
        logger.info("Removed %d duplicate rows", removed)
    return df


# Validate column types
def validate_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct column types."""
    df = df.copy()
    df["productid"] = df["productid"].astype(int)
    df["imageid"] = df["imageid"].astype(int)
    df["prdtypecode"] = df["prdtypecode"].astype(int)
    df["prodtype"] = df["prodtype"].astype(str)
    df["product_designation"] = df["product_designation"].astype(str)
    df["product_description"] = df["product_description"].fillna("").astype(str)
    logger.info("Column types validated")
    return df


# Clean text columns
def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to designation and description."""
    df = df.copy()
    df["designation"] = df["product_designation"].map(clean_text)
    df["description"] = df["product_description"].map(clean_text)
    logger.info("Text columns cleaned")
    return df


# Validate image existence
def validate_images(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Check if each product's image file exists in /data/raw/images/{run_id}/."""
    df = df.copy()
    images_dir = RAW_IMAGES_DIR / run_id

    def image_exists(row):
        filename = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        return (images_dir / filename).exists()

    df["image_exists"] = df.apply(image_exists, axis=1)

    found = df["image_exists"].sum()
    total = len(df)
    logger.info("Image validation: %d / %d found", found, total)
    return df


# Build relative image path for MinIO
def build_image_paths(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Build the relative MinIO path for each image."""
    df = df.copy()

    def build_path(row):
        if not bool(row["image_exists"]):
            return None
        filename = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        return f"{MINIO_IMAGES_PREFIX}{run_id}/{filename}"

    df["path_image_minio"] = df.apply(build_path, axis=1)
    logger.info("Image paths built for batch %s", run_id)
    return df


# List existing images in MinIO (for skip duplicate)
def _get_existing_minio_keys(s3_client, bucket: str, prefix: str) -> set:
    """List all existing object keys under a prefix in MinIO."""
    existing = set()
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            existing.add(obj["Key"])
    return existing


# Upload images to MinIO (skip existing)
def upload_images_to_minio(
    df: pd.DataFrame,
    run_id: str,
    s3_client=None,
    bucket: str = MINIO_BUCKET,
) -> int:
    """
    Upload valid images from /data/raw/images/{run_id}/ to MinIO.
    Skips images already present in MinIO to avoid redundant uploads.
    """
    s3 = s3_client or get_s3_client()
    images_dir = RAW_IMAGES_DIR / run_id

    # Get existing images
    prefix = f"{MINIO_IMAGES_PREFIX}{run_id}/"
    existing_keys = _get_existing_minio_keys(s3, bucket, prefix)

    valid_rows = df[df["image_exists"]]
    uploaded = 0
    skipped = 0

    for _, row in valid_rows.iterrows():
        filename = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        minio_key = f"{prefix}{filename}"

        if minio_key in existing_keys:
            skipped += 1
            continue

        local_path = images_dir / filename
        s3.upload_file(str(local_path), bucket, minio_key)
        uploaded += 1

    logger.info(
        "MinIO images: %d uploaded, %d skipped (already exist), %d total",
        uploaded,
        skipped,
        len(valid_rows),
    )
    return uploaded


# Update category mapping
def update_category_mapping(
    df: pd.DataFrame,
    mapping_path: Path = CATEGORY_MAPPING_PATH,
) -> dict:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing or create empty
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
    else:
        mapping = {}

    # Detect new categories
    batch_categories = df[["prdtypecode", "prodtype"]].drop_duplicates()
    new_count = 0

    for _, row in batch_categories.iterrows():
        code = str(int(row["prdtypecode"]))
        if code not in mapping:
            mapping[code] = row["prodtype"]
            new_count += 1

    # Save updated mapping
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    logger.info("Category mapping updated: %d total, %d new", len(mapping), new_count)
    return mapping


# Save processed CSV
def save_processed_csv(df: pd.DataFrame, run_id: str) -> Path:
    """Save the cleaned DataFrame to /data/processed/{run_id}.csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{run_id}.csv"

    output_cols = [
        "productid",
        "imageid",
        "prdtypecode",
        "prodtype",
        "designation",
        "description",
        "path_image_minio",
        "image_exists",
        "batch_id",
    ]
    df[output_cols].to_csv(output_path, index=False, sep=";")

    logger.info("Processed CSV saved: %s (%d rows)", output_path, len(df))
    return output_path


# Load processed data into PostgreSQL
def load_to_processed_table(df: pd.DataFrame, run_id: str) -> int:
    """
    Insert cleaned rows into products_processed.
    Uses ON CONFLICT DO NOTHING for idempotency.
    Returns the number of rows inserted.
    """
    records = [
        (
            int(row["productid"]),
            int(row["imageid"]),
            int(row["prdtypecode"]),
            str(row["prodtype"]),
            str(row["designation"]),
            str(row["description"]),
            str(row["path_image_minio"])
            if bool(pd.notna(row["path_image_minio"]))
            else None,
            bool(row["image_exists"]),
            run_id,
        )
        for _, row in df.iterrows()
    ]

    insert_sql = """
        INSERT INTO products_processed
            (productid, imageid, prdtypecode, prodtype,
             designation, description,
             path_image_minio, image_exists, batch_id)
        VALUES %s
        ON CONFLICT (productid, batch_id) DO NOTHING
    """

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM products_processed WHERE batch_id = %s",
                (run_id,),
            )
            count_before = cur.fetchone()[0]

            execute_values(cur, insert_sql, records, page_size=1000)

            cur.execute(
                "SELECT COUNT(*) FROM products_processed WHERE batch_id = %s",
                (run_id,),
            )
            count_after = cur.fetchone()[0]

            inserted = count_after - count_before
        conn.commit()
        logger.info(
            "Rows inserted into products_processed: %d / %d", inserted, len(records)
        )
    finally:
        conn.close()

    return inserted
