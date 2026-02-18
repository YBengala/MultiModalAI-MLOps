"""
Airflow DAG – Rakuten Data Pipeline
====================================
Step 0: Cleanup    – Remove /data/tmp/ leftovers
Step 1: Ingestion  – MinIO Sensor -> Download ZIP -> Unzip -> /data/raw/ -> Archive -> Cleanup
Step 1b: Guard     – Skip if batch already fully processed (after archive/cleanup)
Step 2: Transform  – Load raw CSV -> Clean text -> Validate images -> Upload MinIO (skip existing) -> Save processed
Step 3: Quality    – Volume check -> Image match rate -> Designation check
Step 4: Embedding  – Guard -> Generate text + image embeddings -> Append to Parquet
Step 5: MLflow     – Log pipeline metrics and artifacts
Step 6: DVC        – Version data files and commit .dvc to Git (local)
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

# DAG Configuration
MINIO_CONN_ID = "minio_s3"
MINIO_BUCKET = "rakuten-datalake"
MINIO_INCOMING_PREFIX = "incoming/"

default_args = {
    "owner": "ybeng",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="rakuten_ingestion",
    description="End-to-end data pipeline: ingestion, transformation, quality checks, embeddings",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["rakuten", "ingestion", "transformation", "embedding"],
)
def rakuten_ingestion_dag():
    # STEP 0: CLEANUP PREVIOUS RUN
    @task()
    def cleanup_previous_tmp() -> None:
        """Remove /data/tmp/* from previous run."""
        from multimodal_ai.ingestion.ingestion_pipeline import (
            cleanup_previous_tmp as _cleanup,
        )

        _cleanup()

    # STEP 1: INGESTION
    wait_zip = S3KeySensor(
        task_id="wait_zip",
        bucket_name=MINIO_BUCKET,
        bucket_key=f"{MINIO_INCOMING_PREFIX}*.zip",
        wildcard_match=True,
        aws_conn_id=MINIO_CONN_ID,
        poke_interval=300,
        timeout=86400,
        mode="reschedule",
    )

    @task()
    def detect_zip() -> str:
        """Detect and return the ZIP key in MinIO."""
        from multimodal_ai.ingestion.ingestion_pipeline import (
            detect_zip_in_minio,
            get_s3_client,
        )

        s3 = get_s3_client()
        zip_key = detect_zip_in_minio(s3)
        if zip_key is None:
            raise FileNotFoundError("No ZIP found in MinIO incoming")
        return zip_key

    @task()
    def download_zip(zip_key: str) -> str:
        """Download the ZIP from MinIO to /data/tmp/."""
        from multimodal_ai.ingestion.ingestion_pipeline import (
            download_zip_from_minio,
            get_s3_client,
        )

        s3 = get_s3_client()
        local_path = download_zip_from_minio(zip_key, s3)
        return str(local_path)

    @task()
    def extract_and_move(zip_path_str: str) -> dict:
        """Unzip, identify contents, move to /data/raw/."""
        from multimodal_ai.ingestion.ingestion_pipeline import (
            extract_run_id,
            identify_contents,
            move_to_raw,
            unzip_file,
        )

        zip_path = Path(zip_path_str)
        run_id = extract_run_id(zip_path)
        tmp_dir = unzip_file(zip_path)
        csv_path, images = identify_contents(tmp_dir)
        dst_csv, dst_images_dir = move_to_raw(csv_path, images, run_id)

        return {
            "run_id": run_id,
            "csv_path": str(dst_csv),
            "images_dir": str(dst_images_dir),
            "nb_images": len(images),
        }

    @task()
    def archive_zip(zip_key: str) -> str:
        """Archive the ZIP from incoming/ to archivage/ in MinIO."""
        from multimodal_ai.ingestion.ingestion_pipeline import (
            archive_zip_in_minio,
            get_s3_client,
        )

        s3 = get_s3_client()
        archive_key = archive_zip_in_minio(zip_key, s3)
        return archive_key

    @task()
    def cleanup_task(metadata: dict, zip_key: str) -> dict:
        """Remove local tmp and delete ZIP from MinIO incoming."""
        from multimodal_ai.ingestion.ingestion_pipeline import (
            cleanup,
            get_s3_client,
        )

        s3 = get_s3_client()
        cleanup(zip_key=zip_key, s3_client=s3)
        metadata["dt_ingested"] = datetime.now().isoformat()
        return metadata

    # STEP 1b: BATCH GUARD
    @task()
    def check_duplicate_batch(metadata: dict) -> dict:
        """Skip pipeline if batch already fully processed."""
        from multimodal_ai.ingestion.skip_duplicate_batch import (
            is_batch_already_processed,
        )

        if is_batch_already_processed(metadata["run_id"]):
            raise AirflowSkipException(f"Batch {metadata['run_id']} already processed")
        return metadata

    @task()
    def load_to_postgres(metadata: dict) -> dict:
        """Load raw CSV into the products_raw table."""
        from multimodal_ai.ingestion.ingestion_pipeline import load_csv_to_raw_table

        nb_inserted = load_csv_to_raw_table(
            Path(metadata["csv_path"]),
            metadata["run_id"],
        )
        metadata["nb_rows_inserted"] = nb_inserted
        return metadata

    # STEP 2: TRANSFORMATION
    @task()
    def transform_data(metadata: dict) -> dict:
        """
        Load raw CSV, clean text, validate images,
        build image paths, update category mapping.
        """
        from multimodal_ai.transformation.transformation_pipeline import (
            build_image_paths,
            clean_text_columns,
            load_raw_csv,
            remove_duplicates,
            update_category_mapping,
            validate_images,
            validate_types,
        )

        run_id = metadata["run_id"]

        raw_data = load_raw_csv(run_id)
        clean_data = remove_duplicates(raw_data)
        clean_data = validate_types(clean_data)
        clean_data = clean_text_columns(clean_data)
        clean_data = validate_images(clean_data, run_id)
        clean_data = build_image_paths(clean_data, run_id)
        update_category_mapping(clean_data)

        metadata["nb_after_dedup"] = len(clean_data)
        metadata["nb_images_found"] = int(clean_data["image_exists"].sum())
        return metadata

    @task()
    def upload_images(metadata: dict) -> dict:
        """Upload valid images to MinIO (skips already existing)."""
        from multimodal_ai.transformation.transformation_pipeline import (
            build_image_paths,
            get_s3_client,
            load_raw_csv,
            remove_duplicates,
            upload_images_to_minio,
            validate_images,
            validate_types,
        )

        run_id = metadata["run_id"]
        s3 = get_s3_client()
        raw_data = load_raw_csv(run_id)
        clean_data = remove_duplicates(raw_data)
        clean_data = validate_types(clean_data)
        clean_data = validate_images(clean_data, run_id)
        clean_data = build_image_paths(clean_data, run_id)
        nb_uploaded = upload_images_to_minio(clean_data, run_id, s3)
        metadata["nb_images_uploaded"] = nb_uploaded
        return metadata

    # STEP 3: QUALITY CHECKS
    @task()
    def quality_checks(metadata: dict) -> dict:
        """Run quality checks on transformed data."""
        from multimodal_ai.transformation.quality_checks import run_quality_checks
        from multimodal_ai.transformation.transformation_pipeline import (
            build_image_paths,
            clean_text_columns,
            load_raw_csv,
            remove_duplicates,
            validate_images,
            validate_types,
        )

        run_id = metadata["run_id"]
        raw_data = load_raw_csv(run_id)
        clean_data = remove_duplicates(raw_data)
        clean_data = validate_types(clean_data)
        clean_data = clean_text_columns(clean_data)
        clean_data = validate_images(clean_data, run_id)
        clean_data = build_image_paths(clean_data, run_id)

        run_quality_checks(clean_data, run_id)
        metadata["quality_passed"] = True
        return metadata

    @task()
    def save_processed(metadata: dict) -> dict:
        """Save processed CSV and load into PostgreSQL."""
        from multimodal_ai.transformation.transformation_pipeline import (
            build_image_paths,
            clean_text_columns,
            load_raw_csv,
            load_to_processed_table,
            remove_duplicates,
            save_processed_csv,
            validate_images,
            validate_types,
        )

        run_id = metadata["run_id"]
        raw_data = load_raw_csv(run_id)
        clean_data = remove_duplicates(raw_data)
        clean_data = validate_types(clean_data)
        clean_data = clean_text_columns(clean_data)
        clean_data = validate_images(clean_data, run_id)
        clean_data = build_image_paths(clean_data, run_id)

        csv_path = save_processed_csv(clean_data, run_id)
        nb_inserted = load_to_processed_table(clean_data, run_id)

        metadata["processed_csv"] = str(csv_path)
        metadata["nb_processed_inserted"] = nb_inserted
        metadata["dt_processed"] = datetime.now().isoformat()
        return metadata

    # STEP 4: EMBEDDINGS
    @task()
    def check_duplicate_emb(metadata: dict) -> dict:
        """Skip embedding if no new products to embed."""
        from multimodal_ai.features.skip_duplicate_emb import has_new_products

        if not has_new_products(Path(metadata["processed_csv"])):
            raise AirflowSkipException("No new products to embed")
        return metadata

    @task()
    def generate_embeddings(metadata: dict) -> dict:
        """Generate text + image embeddings, save batch Parquet to /data/tmp/."""
        from multimodal_ai.features.build_embeddings import (
            build_batch_parquet,
            filter_valid_products,
            generate_image_embeddings,
            generate_text_embeddings,
            load_processed_csv,
            prepare_text_input,
        )

        run_id = metadata["run_id"]

        processed_data = load_processed_csv(run_id)
        valid_data = filter_valid_products(processed_data)
        valid_data = prepare_text_input(valid_data)
        text_embeddings = generate_text_embeddings(valid_data)
        image_embeddings = generate_image_embeddings(valid_data, run_id)
        batch_path = build_batch_parquet(
            valid_data, text_embeddings, image_embeddings, run_id
        )

        metadata["batch_parquet"] = str(batch_path)
        metadata["nb_embeddings"] = len(valid_data)
        return metadata

    @task()
    def append_embeddings(metadata: dict) -> dict:
        """Append batch embeddings to main Parquet."""
        from multimodal_ai.features.build_embeddings import append_to_embeddings

        batch_path = Path(metadata["batch_parquet"])
        nb_appended = append_to_embeddings(batch_path)

        metadata["nb_appended"] = nb_appended
        metadata["dt_embeddings"] = datetime.now().isoformat()
        return metadata

    # STEP 5: MLFLOW LOGGING
    @task()
    def log_to_mlflow(metadata: dict) -> dict:
        """Log pipeline metrics, parameters and artifacts to MLflow."""
        from multimodal_ai.tracking.mlflow_logger import log_pipeline_run

        mlflow_run_id = log_pipeline_run(metadata)
        metadata["mlflow_run_id"] = mlflow_run_id
        return metadata

    # STEP 6: DVC VERSIONING
    @task()
    def version_data(metadata: dict) -> None:
        """Version data/raw, data/processed, data/embeddings with DVC → MinIO."""
        from multimodal_ai.versioning.dvc_versioning import version_pipeline_data

        version_pipeline_data(metadata["run_id"])

    # ORCHESTRATION

    # Step 0: Clean previous tmp
    clean_tmp = cleanup_previous_tmp()

    # Step 1: Ingestion
    zip_key = detect_zip()
    zip_path = download_zip(zip_key)
    ingestion_metadata = extract_and_move(zip_path)

    # Archive + cleanup TOUJOURS (même si batch dupliqué)
    archived = archive_zip(zip_key)
    cleanup_result = cleanup_task(ingestion_metadata, zip_key)

    # Step 1b: Batch guard (après archive/cleanup)
    checked_metadata = check_duplicate_batch(cleanup_result)

    # Step 1c: Load to PostgreSQL
    checked_metadata = load_to_postgres(checked_metadata)

    # Step 2: Transformation (upload skips existing images in MinIO)
    transform_metadata = transform_data(checked_metadata)
    upload_metadata = upload_images(checked_metadata)

    # Step 3: Quality check -> Save
    quality_metadata = quality_checks(transform_metadata)
    processed_metadata = save_processed(quality_metadata)

    # Step 4: Embedding guard + generation
    processed_metadata = check_duplicate_emb(processed_metadata)
    embedding_metadata = generate_embeddings(processed_metadata)
    final_metadata = append_embeddings(embedding_metadata)

    # Step 5: MLflow logging
    mlflow_metadata = log_to_mlflow(final_metadata)

    # Step 6: DVC versioning
    version_data(mlflow_metadata)

    # Dependencies
    wait_zip >> clean_tmp >> zip_key
    archived >> cleanup_result


rakuten_ingestion_dag()
