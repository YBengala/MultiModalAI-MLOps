"""
Embedding Pipeline :
    - Loads df_model.csv (preprocessed).
    - Encodes text and images using pre-trained models.
    - Fuses embeddings.
    - Saves to processed directory.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from multimodal_ai.config.settings import settings
from multimodal_ai.features.base_text_embedder import BaseTextEmbedder
from multimodal_ai.features.image_encoder_train import ImageEncoderTrain
from multimodal_ai.models.fusion_module import FusionEmbeddings


def build_embeddings(
    csv_path: Path | None = None,
) -> Path:
    """Build, fuse and save embeddings."""

    # --- Paths ---
    csv_path = csv_path or settings.PROCESSED_DATA_DIR / "df_model.csv"
    output_dir = settings.PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = settings.PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    print(f"Dataset: {len(df)} rows, {df['prodtype'].nunique()} classes")

    # --- Fix relative paths ---
    df["path_image"] = df["path_image"].apply(
        lambda p: str((settings.PROJECT_ROOT / p.lstrip("./")).resolve())
    )
    print(f"Sample path: {df['path_image'].iloc[0]}")

    # --- Labels ---
    le = LabelEncoder()
    y = np.asarray(le.fit_transform(df["prodtype"].values))

    # --- Text embeddings ---
    print(f"Encoding text with {settings.TEXT_MODEL_NAME}...")
    text_encoder = BaseTextEmbedder()
    texts = df["text"].fillna("").tolist()
    text_emb = text_encoder.encode_text(texts)
    print(f"Text embeddings: {text_emb.shape}")

    # --- Image embeddings ---
    print(f"Encoding images with {settings.IMAGE_MODEL_NAME}...")
    image_encoder = ImageEncoderTrain()
    img_emb = image_encoder.image_train_encodings(df, path_column="path_image")
    print(f"Image embeddings: {img_emb.shape}")

    # --- Fusion ---
    print("Fusing embeddings...")
    fuser = FusionEmbeddings()
    combined = fuser.fuse_embeddings(img_emb, text_emb)
    print(f"Combined embeddings: {combined.shape}")

    # --- Save ---
    np.save(output_dir / "embeddings_text.npy", text_emb)
    np.save(output_dir / "embeddings_image.npy", img_emb)
    np.save(output_dir / "embeddings_combined.npy", combined)
    np.save(output_dir / "labels.npy", y)
    joblib.dump(le, models_dir / "label_encoder.joblib")

    print(f"\nDone — saved to {output_dir}:")
    print(f"  text:     {text_emb.shape}")
    print(f"  image:    {img_emb.shape}")
    print(f"  combined: {combined.shape}")
    print(f"  labels:   {y.shape}")
    print(f"  encoder:  {models_dir / 'label_encoder.joblib'}")

    return output_dir


if __name__ == "__main__":
    build_embeddings()
