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
from typing import Any, Dict

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from multimodal_ai.config.settings import settings
from multimodal_ai.models.fusion_module import MultimodalMLP
from multimodal_ai.utils.callbacks import EarlyStopping


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
    cfg: Dict[str, Any] | None = None,
) -> float:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    params = {
        "batch_size": settings.batch_size,
        "learning rate": 1e-3,
        "epochs": 50,
        "dropout": 0.3,
        "patience_es": 7,
        "patience_lr": 3,
        "min_delta": 1e-4,
        "seed": 42,
        "hidden_l1": 2048,
        "hidden_l2": 1024,
        "activation": "ReLU",
    }

    if cfg:
        params.update(cfg)

    if trial:
        print(f"Optuna_Trial : {trial.number}: hyperparameters.")
        params["learning rate"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        params["dropout"] = trial.suggest_float("dropout", 0.2, 0.6)
        params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
        params["hidden_l1"] = trial.suggest_int("hidden_l1", 1024, 4096, step=256)
        params["hidden_l2"] = trial.suggest_int("hidden_l2", 256, 1024, step=128)
        params["activation"] = trial.suggest_categorical(
            "activation", ["ReLU", "GELU", "SiLU"]
        )

    activation_map = {"ReLU": nn.ReLU(), "GELU": nn.GELU(), "SiLU": nn.SiLU()}
    activation_cls = activation_map[params["activation"]]

    DEVICE = torch.device(settings.DEFAULT_DEVICE)
    print(f"\n Starting Pipeline on {DEVICE} | Params: {params}")

    # data loading
    data_path = settings.PROCESSED_DATA_DIR
    X = np.load(data_path / "embeddings_combined_scaled.npy")
    df = pd.read_csv(data_path / "df_model.csv", sep=";")

    le = LabelEncoder()
    y = le.fit_transform(df["prodtype"].values)

    models_dir = settings.PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(le, models_dir / "label_encoder.joblib")

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
    )
    val_loader = DataLoader(
        MultimodalDataset(X_val, y_val),
        batch_size=params["batch_size"],
        shuffle=False,
    )

    # model construction
    num_classes_safe = len(np.unique(np.asarray(y)))

    model = MultimodalMLP(
        input_dim=X.shape[1],
        num_classes=num_classes_safe,
        hidden_l1=params["hidden_l1"],
        hidden_l2=params["hidden_l2"],
        dropout_rate=params["dropout"],
        fn_activation=activation_cls,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=params["learning rate"])
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

    print(f"Training started : {params['epochs']} epochs...")

    val_acc = 0.0

    for epoch in range(params["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
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
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(y_val)
        val_acc = accuracy_score(all_labels, all_preds)

        print(
            f"Ep {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}"
        )

        if trial:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                print(f"✂️ Trial pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        # Callbacks
        scheduler.step(avg_val_loss)
        es(avg_val_loss, model)

        if es.early_stop:
            break

    print("\nRestoring best model weights from memory...")
    es.load_best_model(model)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"model_fusion_{timestamp}.pth"
    local_path = models_dir / model_filename

    # Save best model
    torch.save(model.state_dict(), local_path)
    print(f"Model saved at : {local_path}")

    return val_acc


if __name__ == "__main__":
    args = default_args()
    manual_config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout,
    }
    train_pipeline(cfg=manual_config)
