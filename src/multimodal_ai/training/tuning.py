"""
Hyperparameter Tuning Pipeline :
    - Optimization by Optuna.
"""

from __future__ import annotations

import joblib
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from multimodal_ai.config.settings import settings
from multimodal_ai.training.train import set_seed, train_pipeline


def objective(trial: optuna.Trial) -> float:
    try:
        return train_pipeline(trial=trial)
    except optuna.exceptions.TrialPruned:
        raise
    except KeyboardInterrupt:
        trial.study.stop()
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("-inf")


if __name__ == "__main__":
    set_seed(42)

    print("Starting Hyperparameter Tuning.")

    storage = f"sqlite:///{settings.PROJECT_ROOT / 'models' / 'optuna.db'}"
    study = optuna.create_study(
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        study_name="Rakuten_Fusion_V2",
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=15,
        ),
        sampler=TPESampler(seed=42),
    )

    try:
        print("Optimizing : 20 trials")
        study.optimize(objective, n_trials=20)
    except KeyboardInterrupt:
        print("\nInterrupted: Saving current best results")

    print("\nOptimization Finished")
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
