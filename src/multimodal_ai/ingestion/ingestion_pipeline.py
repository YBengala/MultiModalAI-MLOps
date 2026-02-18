"""
Detects a ZIP in MinIO (incoming/), extracts CSV + images,
stores them in /data/raw/ and loads the products_raw table (PostgreSQL).
"""

import logging
import os
import shutil
import zipfile
from pathlib import Path

import boto3
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

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
MINIO_INCOMING_PREFIX = "incoming/"
MINIO_ARCHIVE_PREFIX = "archivage/"

RAW_DIR = Path("/data/raw")
RAW_IMAGES_DIR = Path("/data/raw/images")
TMP_DIR = Path("/data/tmp")

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


# Detect ZIP in MinIO incoming bucket
def detect_zip_in_minio(
    s3_client=None,
    bucket: str = MINIO_BUCKET,
    prefix: str = MINIO_INCOMING_PREFIX,
) -> str | None:
    """
    List objects in rakuten-datalake/incoming/ and return the key of the first ZIP found, or None.
    """
    s3 = s3_client or get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    for obj in response.get("Contents", []):
        if obj["Key"].endswith(".zip"):
            logger.info("ZIP detected in MinIO: %s/%s", bucket, obj["Key"])
            return obj["Key"]

    logger.info("No ZIP found in %s/%s", bucket, prefix)
    return None


# Download ZIP to local data/tmp
def download_zip_from_minio(
    zip_key: str,
    s3_client=None,
    bucket: str = MINIO_BUCKET,
    tmp_dir: Path = TMP_DIR,
) -> Path:
    """Download ZIP from MinIO to /data/tmp/."""
    s3 = s3_client or get_s3_client()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    local_path = tmp_dir / Path(zip_key).name
    s3.download_file(bucket, zip_key, str(local_path))

    logger.info("ZIP downloaded: %s -> %s", zip_key, local_path)
    return local_path


# Extract run_id from ZIP filename
def extract_run_id(zip_path: Path) -> str:
    """Derive the batch_id from the ZIP filename (without extension)."""
    run_id = zip_path.stem
    logger.info("Run ID: %s", run_id)
    return run_id


# Unzip into temporary directory
def unzip_file(zip_path: Path, tmp_dir: Path = TMP_DIR) -> Path:
    """Extract ZIP contents into a temporary subdirectory."""
    unzip_dir = tmp_dir / "unzip"
    if unzip_dir.exists():
        shutil.rmtree(unzip_dir)
    unzip_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(unzip_dir)

    logger.info("ZIP extracted to %s", unzip_dir)
    return unzip_dir


# Identify CSV and images in extracted directory
def identify_contents(tmp_dir: Path) -> tuple[Path, list[Path]]:
    """
    Scan the extracted directory and return:
      - path to the CSV file
      - list of image file paths
    """
    csv_files = list(tmp_dir.rglob("*.csv"))
    if len(csv_files) != 1:
        raise FileNotFoundError(f"Expected 1 CSV, found {len(csv_files)} in {tmp_dir}")

    image_extensions = {".jpg", ".jpeg", ".png"}
    images = [f for f in tmp_dir.rglob("*") if f.suffix.lower() in image_extensions]

    logger.info("CSV found: %s | Images: %d", csv_files[0].name, len(images))
    return csv_files[0], images


# Move files to /data/raw/
def move_to_raw(
    csv_path: Path,
    images: list[Path],
    run_id: str,
    raw_dir: Path = RAW_DIR,
    raw_images_dir: Path = RAW_IMAGES_DIR,
) -> tuple[Path, Path]:
    """
    Copy CSV and images to /data/raw/ with batch naming.
    Returns (destination CSV path, destination images directory).
    """
    # CSV
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = raw_dir / f"{run_id}.csv"
    shutil.copy2(csv_path, dst_csv)
    logger.info("CSV copied -> %s", dst_csv)

    # Images
    dst_images_dir = raw_images_dir / f"{run_id}"
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        shutil.copy2(img, dst_images_dir / img.name)

    logger.info("Images copied -> %s (%d files)", dst_images_dir, len(images))
    return dst_csv, dst_images_dir


# Load raw CSV into PostgreSQL (products_raw table)
def load_csv_to_raw_table(csv_path: Path, run_id: str) -> int:
    """
    Read the raw CSV and insert rows into products_raw.
    Uses ON CONFLICT DO NOTHING for idempotency.
    Returns the number of rows inserted.
    """
    df = pd.read_csv(csv_path, sep=";")
    cols = {
        "productid",
        "imageid",
        "prdtypecode",
        "prodtype",
        "product_designation",
        "product_description",
    }
    missing = cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df.columns = df.columns.str.lower().str.strip()
    df["product_description"] = df["product_description"].fillna("")
    records = [
        (
            int(row["productid"]),
            int(row["imageid"]),
            int(row["prdtypecode"]),
            str(row["prodtype"]),
            str(row["product_designation"]),
            str(row["product_description"]),
            run_id,
        )
        for _, row in df.iterrows()
    ]

    insert_sql = """
        INSERT INTO products_raw
            (productid, imageid, prdtypecode, prodtype,
             product_designation, product_description, batch_id)
        VALUES %s
        ON CONFLICT (productid, batch_id) DO NOTHING
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM products_raw WHERE batch_id = %s", (run_id,)
            )
            count_before = cur.fetchone()[0]

            execute_values(cur, insert_sql, records, page_size=1000)

            cur.execute(
                "SELECT COUNT(*) FROM products_raw WHERE batch_id = %s", (run_id,)
            )
            count_after = cur.fetchone()[0]

            inserted = count_after - count_before
        conn.commit()
        logger.info("Rows inserted into products_raw: %d / %d", inserted, len(records))
    finally:
        conn.close()

    return inserted


# Archive ZIP in MinIO
def archive_zip_in_minio(
    zip_key: str,
    s3_client=None,
    bucket: str = MINIO_BUCKET,
    archive_prefix: str = MINIO_ARCHIVE_PREFIX,
) -> str:
    """
    Copy the ZIP from incoming/ to archivage/
    Returns the destination key.
    """
    s3 = s3_client or get_s3_client()

    filename = Path(zip_key).name
    archive_key = f"{archive_prefix}{filename}"

    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": zip_key},
        Key=archive_key,
    )

    logger.info("ZIP archived: %s -> %s", zip_key, archive_key)
    return archive_key


# Cleanup /data/tmp/ from previous run
def cleanup_previous_tmp(tmp_dir: Path = TMP_DIR) -> None:
    """Remove leftover tmp directory from previous run."""
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        logger.info("Previous tmp directory cleaned: %s", tmp_dir)


# Post-ingestion cleanup (MinIO only)
def cleanup(
    zip_key: str,
    s3_client=None,
    bucket: str = MINIO_BUCKET,
) -> None:
    """Delete ZIP from MinIO incoming after archiving."""
    s3 = s3_client or get_s3_client()
    s3.delete_object(Bucket=bucket, Key=zip_key)
    logger.info("ZIP deleted from MinIO: %s/%s", bucket, zip_key)
