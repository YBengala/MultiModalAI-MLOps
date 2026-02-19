"""
FastAPI inference service for the Rakuten Multimodal Fusion Model.

Loads the Production model from MLflow Model Registry at startup.
Exposes a /predict endpoint that accepts a product designation, description,
and image, runs them through the text and image encoders, and returns
the predicted category.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow.pytorch
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from multimodal_ai.config.settings import settings
from multimodal_ai.features.image_encoder_infer import ImageEncoderInfer
from multimodal_ai.features.text_encoder_infer import TextEncoderInfer
from multimodal_ai.transformation.text_cleaner import input_text_infer

logger = logging.getLogger(__name__)

# Global state loaded at startup
_model: torch.nn.Module | None = None
_text_encoder: TextEncoderInfer | None = None
_image_encoder: ImageEncoderInfer | None = None
_idx_to_label: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and encoders once at startup, release at shutdown."""
    global _model, _text_encoder, _image_encoder, _idx_to_label

    logger.info("Loading encoders...")
    _text_encoder = TextEncoderInfer()
    _image_encoder = ImageEncoderInfer()

    logger.info("Loading Production model from MLflow registry...")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.MINIO_ROOT_USER)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.MINIO_ROOT_PASSWORD)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.MLFLOW_S3_ENDPOINT_URL)
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    _model = mlflow.pytorch.load_model("models:/Rakuten_Multimodal_Fusion/Production")
    _model.eval()

    # Load class mapping from MLflow artifacts
    client = mlflow.tracking.MlflowClient()
    prod_versions = client.get_latest_versions(
        "Rakuten_Multimodal_Fusion", stages=["Production"]
    )
    if prod_versions:
        run_id = prod_versions[0].run_id
        local_path = client.download_artifacts(run_id, "class_mapping.json")
        with open(local_path) as f:
            _idx_to_label = json.load(f)
        logger.info("Class mapping loaded: %d classes", len(_idx_to_label))

    logger.info("Inference service ready.")
    yield

    _model = None
    _text_encoder = None
    _image_encoder = None
    _idx_to_label = {}


app = FastAPI(
    title="Rakuten Multimodal Inference",
    description="Predicts product category from text + image.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    predicted_class_index: int
    predicted_label: str
    confidence: float
    top5: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    designation: str = Form(..., description="Product designation"),
    description: str | None = Form(None, description="Product description (optional)"),
    image: UploadFile = File(..., description="Product image"),
) -> PredictionResponse:
    """
    Predict the product category from text fields and image.

    - designation: product title / designation (required)
    - description: product description (optional)
    - image: product image file (JPEG/PNG)
    """
    if _model is None or _text_encoder is None or _image_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build unified text input — same preprocessing as training pipeline
    text = input_text_infer(designation, description)

    # Encode inputs
    image_bytes = await image.read()
    text_emb = _text_encoder.encode_text_infer(text)  # (768,)
    image_emb = _image_encoder.encode_image_bytes(image_bytes)  # (384,)

    # Fuse: image first, then text (same order as training)
    fused = np.concatenate([image_emb, text_emb], axis=0).astype(np.float32)
    x = torch.FloatTensor(fused).unsqueeze(0)  # (1, 1152)

    # Inference
    with torch.no_grad():
        logits = _model(x)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1)[0]  # (num_classes,)

    pred_idx = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())
    pred_label = _idx_to_label.get(str(pred_idx), f"class_{pred_idx}")

    # Top-5 classes
    top5_indices = probs.topk(min(5, len(probs))).indices.tolist()
    top5 = [
        {
            "class_index": i,
            "label": _idx_to_label.get(str(i), f"class_{i}"),
            "confidence": float(probs[i].item()),
        }
        for i in top5_indices
    ]

    return PredictionResponse(
        predicted_class_index=pred_idx,
        predicted_label=pred_label,
        confidence=confidence,
        top5=top5,
    )
