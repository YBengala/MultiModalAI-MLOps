"""
Hyperparameter Tuning Pipeline :
    - Optimization by Optuna.
"""

from __future__ import annotations

import joblib
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from multimodal_ai.config.settings import settings
from multimodal_ai.training.train import train_pipeline


def objective(trial: optuna.Trial) -> float:
    try:
        return train_pipeline(trial=trial)
    except KeyboardInterrupt:
        trial.study.stop()
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("-inf")


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Starting Hyperparameter Tuning.")

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        study_name="Rakuten_Fusion_Optimization_Local",
        pruner=pruner,
        sampler=TPESampler(seed=42),
    )

    # optimization exec
    try:
        print("Optimizing : 10 trials")
        study.optimize(objective, n_trials=10)
    except KeyboardInterrupt:
        print("\n Interrupted: Saving current best results")

    # saves best hyperparameters
    print("\n Optimization Finished")
    trial_completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if len(trial_completed) > 0:
        print(f"Best Accuracy: {study.best_value:.4f}")
        print(f"Best Params: {study.best_params}")

        models_dir = settings.PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True, parents=True)

        joblib.dump(study.best_params, models_dir / "best_hyperparams.joblib")
        print(f"Best params at : {models_dir}")

    else:
        print("No trials completed.")
