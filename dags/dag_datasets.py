from airflow.datasets import Dataset

EMBEDDINGS_DATASET = Dataset("s3://rakuten-datalake/embeddings/embeddings.parquet")
