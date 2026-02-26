"""
Training pipeline for Multimodal Fusion Model (Rakuten) :
    - Loads pre-computed embeddings (features) and labels.
    - Computes class weights for imbalanced datasets.
    - Optuna hyperparameter tuning.
    - Early Stopping + Learning Rate Scheduling.
    - Saves best model.
    - MLflow tracking (training runs + Optuna trials).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import logging
import os
import random
import subprocess
from typing import Any

import mlflow
import numpy as np
import optuna
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from multimodal_ai.config.settings import settings
from multimodal_ai.models.fusion_module import MultimodalMLP
from multimodal_ai.tracking.mlflow_logger import _get_cumulative_stats
from multimodal_ai.training.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultimodalDataset(Dataset):
    def __init__(self, features: Any, labels: Any):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal Training Pipeline")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument(
        "--batch_size", type=int, default=settings.batch_size, help="Batch Size"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout Rate")
    return parser.parse_args()


def train_pipeline(
    trial: optuna.Trial | None = None,
    cfg: dict[str, Any] | None = None,
    mlflow_parent_run_id: str | None = None,
    data_run_id: str | None = None,
) -> float:
    params = {
        "batch_size": settings.batch_size,
        "learning_rate": 1e-3,
        "epochs": 50,
        "dropout": 0.3,
        "weight_decay": 0.05,
        "patience_es": 7,
        "patience_lr": 3,
        "min_delta": 1e-4,
        "seed": 42,
        "hidden_l1": 1024,
        "hidden_l2": 512,
        "hidden_l3": 256,
        "activation": "SiLU",
    }

    if cfg:
        params.update(cfg)

    set_seed(params["seed"])

    if trial:
        logger.info("Optuna Trial %d: sampling hyperparameters.", trial.number)
        params["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        params["dropout"] = trial.suggest_float("dropout", 0.2, 0.6)
        params["weight_decay"] = trial.suggest_float("wd", 0.01, 0.1, log=True)
        params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
        params["hidden_l1"] = trial.suggest_int("hidden_l1", 512, 2048, step=256)
        params["hidden_l2"] = trial.suggest_int("hidden_l2", 256, 1024, step=128)
        params["hidden_l3"] = trial.suggest_int("hidden_l3", 128, 512, step=64)
        params["activation"] = trial.suggest_categorical(
            "activation", ["ReLU", "GELU", "SiLU"]
        )

    activation_map: dict[str, type[nn.Module]] = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
    }
    activation_fn = activation_map[params["activation"]]()

    DEVICE = torch.device(settings.DEFAULT_DEVICE)
    use_cuda = DEVICE.type == "cuda"
    print(f"\nStarting pipeline on {DEVICE} | Params: {params}")

    # Data loading
    embeddings_path = settings.DATA_DIR / "embeddings" / "embeddings.parquet"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    df = pq.read_table(embeddings_path).to_pandas()
    X_text = np.stack(df["embedding_text"].values)
    X_image = np.stack(df["embedding_image"].values)
    X = np.concatenate([X_image, X_text], axis=1).astype(np.float32)

    # Category mapping: prdtypecode -> label (stable source of truth)
    category_mapping_path = settings.PROCESSED_DATA_DIR / "category_mapping.json"
    if not category_mapping_path.exists():
        raise FileNotFoundError(
            f"category_mapping.json not found: {category_mapping_path}"
        )
    with open(category_mapping_path) as f:
        category_mapping: dict[str, str] = json.load(f)

    # Stable encoding: sorted order of known codes from category_mapping.
    # Guarantees that class index N maps to the same category on every run,
    all_known_codes = sorted(int(c) for c in category_mapping.keys())
    le = LabelEncoder()
    le.fit(all_known_codes)
    y = le.transform(df["prdtypecode"].astype(int).values)

    # label index -> label
    idx_to_label = {
        i: category_mapping[str(code)] for i, code in enumerate(le.classes_)
    }

    models_dir = settings.PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=params["seed"], stratify=y
    )

    # imbalance management
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(np.asarray(y_train)), y=np.asarray(y_train)
    )
    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    train_loader = DataLoader(
        MultimodalDataset(X_train, y_train),
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0,
        persistent_workers=use_cuda,
    )
    val_loader = DataLoader(
        MultimodalDataset(X_val, y_val),
        batch_size=params["batch_size"],
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0,
        persistent_workers=use_cuda,
    )

    # Model
    num_classes = len(np.unique(np.asarray(y)))

    model = MultimodalMLP(
        input_dim=X.shape[1],
        num_classes=num_classes,
        hidden_l1=params["hidden_l1"],
        hidden_l2=params["hidden_l2"],
        hidden_l3=params["hidden_l3"],
        dropout_rate=params["dropout"],
        fn_activation=activation_fn,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=params["patience_lr"],
    )

    es = EarlyStopping(
        patience=params["patience_es"],
        min_delta=params["min_delta"],
    )

    # Best model checkpoint based on val_f1_macro
    best_f1 = 0.0
    best_model_state = None
    current_run_id: str | None = None

    print(f"Training started: {params['epochs']} epochs...")
    val_acc = 0.0

    # MLflow setup — credentials for upload artifacts to MinIO
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.MINIO_ROOT_USER)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.MINIO_ROOT_PASSWORD)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.MLFLOW_S3_ENDPOINT_URL)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    if trial is None:
        mlflow.set_experiment(settings.MLFLOW_TRAINING_EXPERIMENT_NAME)
        run_name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_run = mlflow.start_run(run_name=run_name)
    else:
        mlflow.set_experiment(settings.MLFLOW_TUNING_EXPERIMENT_NAME)
        mlflow_run = mlflow.start_run(
            run_name=f"trial_{trial.number}",
            parent_run_id=mlflow_parent_run_id,
            nested=True,
        )

    with mlflow_run:
        mlflow.log_params(
            {
                "learning_rate": params["learning_rate"],
                "batch_size": params["batch_size"],
                "epochs": params["epochs"],
                "dropout": params["dropout"],
                "weight_decay": params["weight_decay"],
                "hidden_l1": params["hidden_l1"],
                "hidden_l2": params["hidden_l2"],
                "hidden_l3": params["hidden_l3"],
                "activation": params["activation"],
                "seed": params["seed"],
            }
        )
        mlflow.set_tags(
            {
                "pipeline": "rakuten_training" if trial is None else "rakuten_tuning",
                "device": settings.DEFAULT_DEVICE,
            }
        )
        mlflow.log_param("num_classes", len(le.classes_))
        mlflow.log_dict(idx_to_label, "class_mapping.json")

        # Training data statistics
        if data_run_id:
            mlflow.set_tag("data_run_id", data_run_id)
        try:
            repo_dir = os.environ.get("REPO_DIR", ".")
            git_sha = (
                subprocess.check_output(
                    ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            mlflow.set_tag("git_sha", git_sha)
        except Exception:
            pass
        mlflow.log_params(
            {
                "data_nb_samples": len(df),
                "data_nb_classes": num_classes,
                "data_nb_train": len(X_train),
                "data_nb_val": len(X_val),
            }
        )
        cumul = _get_cumulative_stats()
        if cumul:
            mlflow.log_params(
                {
                    "data_total_products": cumul["total_products"],
                    "data_imbalance_ratio": cumul["imbalance_ratio"],
                }
            )

        for epoch in range(params["epochs"]):
            # Train
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            avg_train_loss = train_loss / len(y_train)

            # Validation
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    all_preds.append(outputs.argmax(1).cpu())
                    all_labels.append(labels.cpu())

            all_preds_np = torch.cat(all_preds).numpy()
            all_labels_np = torch.cat(all_labels).numpy()
            avg_val_loss = val_loss / len(y_val)
            val_acc = accuracy_score(all_labels_np, all_preds_np)
            val_f1 = f1_score(
                all_labels_np, all_preds_np, average="macro", zero_division=0
            )

            print(
                f"Ep {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} "
                f"| Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                    "val_f1_macro": val_f1,
                },
                step=epoch,
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())

            if trial:
                trial.report(val_f1, epoch)
                if trial.should_prune():
                    print(f"Trial pruned at epoch {epoch}")
                    mlflow.set_tag("pruned", "true")
                    raise optuna.exceptions.TrialPruned()

            scheduler.step(avg_val_loss)
            es(avg_val_loss, model)

            if es.early_stop:
                break

        print("Restoring best model weights (best val_f1_macro)...")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        val_f1 = best_f1

        mlflow.log_metrics(
            {
                "final_val_acc": val_acc,
                "final_val_f1_macro": val_f1,
                "best_val_loss": float(es.best_loss),
            }
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        model_filename = f"model_fusion_{timestamp}.pth"
        local_path = models_dir / model_filename

        # save best model
        if trial is None:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "params": params,
                    "input_dim": X.shape[1],
                    "num_classes": num_classes,
                    "val_accuracy": val_acc,
                    "val_f1_macro": val_f1,
                    "timestamp": timestamp,
                    "idx_to_label": idx_to_label,
                },
                local_path,
            )
            print(f"Model saved at: {local_path}")

            # Log model artifact (without registering yet — run must be closed first)
            current_run_id = mlflow.active_run().info.run_id
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=f"model_{data_run_id}" if data_run_id else "model",
            )

    # Register model and auto-promote after the run is fully closed
    if trial is None and current_run_id is not None:
        client = mlflow.tracking.MlflowClient()

        # Ensure the registered model exists (first run after reset)
        try:
            client.create_registered_model("Rakuten_Multimodal_Fusion")
        except Exception:
            pass  # Already exists

        # Register the model version from the closed run
        artifact_path = f"model_{data_run_id}" if data_run_id else "model"
        model_uri = f"runs:/{current_run_id}/{artifact_path}"
        mv = client.create_model_version(
            name="Rakuten_Multimodal_Fusion",
            source=model_uri,
            run_id=current_run_id,
        )
        print(f"Model registered: version {mv.version}")

        # Compare with current Production version
        prod_versions = client.get_latest_versions(
            "Rakuten_Multimodal_Fusion", stages=["Production"]
        )
        best_prod_f1 = 0.0
        if prod_versions:
            try:
                prod_run = client.get_run(prod_versions[0].run_id)
                best_prod_f1 = float(
                    prod_run.data.metrics.get("final_val_f1_macro", 0.0)
                )
            except Exception:
                best_prod_f1 = 0.0

        if val_f1 > best_prod_f1:
            # Archive previous Production version
            for pv in prod_versions:
                client.transition_model_version_stage(
                    name="Rakuten_Multimodal_Fusion",
                    version=pv.version,
                    stage="Archived",
                )
            # Promote new version to Production
            client.transition_model_version_stage(
                name="Rakuten_Multimodal_Fusion",
                version=mv.version,
                stage="Production",
            )
            print(
                f"Model v{mv.version} promoted to Production "
                f"(F1: {val_f1:.4f} > {best_prod_f1:.4f})"
            )

            # Delete Archived versions to keep the registry clean
            all_versions = client.search_model_versions(
                "name='Rakuten_Multimodal_Fusion'"
            )
            for v in all_versions:
                if v.current_stage == "Archived":
                    client.delete_model_version("Rakuten_Multimodal_Fusion", v.version)
                    print(f"Deleted archived version {v.version}")
        else:
            print(
                f"Model v{mv.version} not promoted "
                f"(F1: {val_f1:.4f} <= Production: {best_prod_f1:.4f})"
            )

    return val_f1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = default_args()
    manual_config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
    }
    train_pipeline(cfg=manual_config)
