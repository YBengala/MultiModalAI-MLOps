"""
Training pipeline for Multimodal Fusion Model (Rakuten) :
    - Loads pre-computed embeddings (features) and labels.
    - Computes class weights for imbalanced datasets.
    - Optuna hyperparameter tuning.
    - Early Stopping + Learning Rate Scheduling.
    - Saves best model.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import random
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from multimodal_ai.config.settings import settings
from multimodal_ai.models.fusion_module import MultimodalMLP
from multimodal_ai.training.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Reproducibility — une seule source de vérité."""
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

    # data loading
    data_path = settings.PROCESSED_DATA_DIR
    models_dir = settings.PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    X = np.load(data_path / "embeddings_combined.npy", mmap_mode="r")
    y = np.load(data_path / "labels.npy")

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

    print(f"Training started: {params['epochs']} epochs...")
    val_acc = 0.0

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

        print(
            f"Ep {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}"
        )

        if trial:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        scheduler.step(avg_val_loss)
        es(avg_val_loss, model)

        if es.early_stop:
            break

    print("Restoring best model weights...")
    es.load_best_model(model)

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
                "timestamp": timestamp,
            },
            local_path,
        )
        print(f"Model saved at: {local_path}")

    return val_acc


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
