import io
import os

import boto3
from botocore.exceptions import ClientError
from PIL import Image


def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://rakuten-minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def fetch_image(path_image_minio: str, bucket: str = "rakuten-datalake") -> Image.Image | None:
    """Fetch an image from MinIO and return a PIL Image, or None on error."""
    try:
        client = get_minio_client()
        response = client.get_object(Bucket=bucket, Key=path_image_minio)
        image_bytes = response["Body"].read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except ClientError:
        return None
    except Exception:
        return None
