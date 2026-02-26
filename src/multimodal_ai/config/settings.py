import os
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Settings(BaseSettings):
    """
    Global configuration for the entire project (training + inference + API).
      - paths
      - device settings
      - batch sizes
      - model names
      - environment variables
    """

    # ==========================
    # PROJECT ROOT
    # ==========================
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

    # ==========================
    # DATA DIRECTORIES
    # ==========================
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    IMAGES_DIR: Path = DATA_DIR / "images"

    # ==========================
    # TEXT EMBEDDING SETTINGS
    # ==========================
    TEXT_MODEL_NAME: str = "OrdalieTech/Solon-embeddings-base-0.1"
    TEXT_NORMALIZE: bool = True

    # ==========================
    # IMAGE EMBEDDING SETTINGS
    # ==========================
    IMAGE_MODEL_NAME: str = "efficientvit_b2.r288_in1k"
    IMAGE_NORMALIZE: bool = True
    IMAGE_SIZE: int = 224

    # ==========================
    # EMBEDDING DIMENSIONS
    # ==========================
    TEXT_EMBEDDING_DIM: int = 768  # Solon output dim
    IMAGE_EMBEDDING_DIM: int = 384  # EfficientViT output dim

    @property
    def fusion_dim(self) -> int:
        return self.TEXT_EMBEDDING_DIM + self.IMAGE_EMBEDDING_DIM

    # ==========================
    # DEVICE & BATCH CONFIG
    # ==========================
    DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE_CPU: int = 64
    BATCH_SIZE_GPU: int = 128

    @property
    def batch_size(self) -> int:
        return self.get_batch_size(self.DEFAULT_DEVICE)

    def get_batch_size(self, device: str) -> int:
        return self.BATCH_SIZE_GPU if device == "cuda" else self.BATCH_SIZE_CPU

    # ==========================
    # MLFLOW SETTINGS
    # ==========================
    MLFLOW_EXPERIMENT_NAME: str = "Rakuten_Multimodal"
    MLFLOW_TRAINING_EXPERIMENT_NAME: str = "Rakuten_Multimodal_Training"
    MLFLOW_TUNING_EXPERIMENT_NAME: str = "Rakuten_Multimodal_Tuning"
    MLFLOW_MONITORING_EXPERIMENT_NAME: str = "Rakuten_Multimodal_Monitoring"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # ==========================
    # INFERENCE API
    # ==========================
    INFERENCE_API_URL: str = "http://localhost:8000"

    # ==========================
    # MINIO SETTINGS
    # ==========================
    MINIO_ROOT_USER: str = Field(..., min_length=1)
    MINIO_ROOT_PASSWORD: str = Field(..., min_length=1)
    MLFLOW_S3_ENDPOINT_URL: str = "http://localhost:9000"

    # ==========================
    # DATABASE
    # ==========================
    RAKUTEN_DB_USER: str = Field(..., min_length=1)
    RAKUTEN_DB_PASSWORD: str = Field(..., min_length=1)
    RAKUTEN_DB_NAME: str = "rakuten_db"
    RAKUTEN_DB_HOST: str = "localhost"
    RAKUTEN_DB_PORT: int = 5433

    @property
    def RAKUTEN_DB_URI(self) -> str:
        return f"postgresql://{self.RAKUTEN_DB_USER}:{self.RAKUTEN_DB_PASSWORD}@{self.RAKUTEN_DB_HOST}:{self.RAKUTEN_DB_PORT}/{self.RAKUTEN_DB_NAME}"

    @property
    def OPTUNA_STORAGE_URI(self) -> str:
        return f"postgresql://{self.RAKUTEN_DB_USER}:{self.RAKUTEN_DB_PASSWORD}@{self.RAKUTEN_DB_HOST}:{self.RAKUTEN_DB_PORT}/{self.RAKUTEN_DB_NAME}"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()  # pyright: ignore

if __name__ == "__main__":
    print("\n=== Loading configuration ===")

    # Display all variables
    for key, value in settings.model_dump().items():
        print(f"{key}: {value}")
