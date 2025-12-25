import os
from pathlib import Path

import torch
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
    TEXT_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    TEXT_NORMALIZE: bool = True

    # ==========================
    # IMAGE EMBEDDING SETTINGS
    # ==========================
    IMAGE_MODEL_NAME: str = "efficientvit_b2.r288_in1k"
    IMAGE_NORMALIZE: bool = True
    IMAGE_SIZE: int = 224

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
    MLFLOW_DIR: Path = PROJECT_ROOT / "mlruns"

    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        return f"file://{self.MLFLOW_DIR}"


settings = Settings()
