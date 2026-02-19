"""
Hyperparameter Tuning Pipeline :
    - Optimization by Optuna.
    - MLflow tracking: parent run (study) + nested child runs (trials).
"""

from __future__ import annotations

import joblib
import mlflow
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from multimodal_ai.config.settings import settings
from multimodal_ai.training.train import set_seed, train_pipeline

_TUNING_PARENT_RUN_ID: str | None = None
_TUNING_DATA_RUN_ID: str | None = None


def objective(trial: optuna.Trial) -> float:
    try:
        return train_pipeline(
            trial=trial,
            mlflow_parent_run_id=_TUNING_PARENT_RUN_ID,
            data_run_id=_TUNING_DATA_RUN_ID,
        )
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

    storage = settings.OPTUNA_STORAGE_URI
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

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_TUNING_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="study_Rakuten_Fusion_V2") as parent_run:
        mlflow.set_tags(
            {"pipeline": "rakuten_tuning", "study_name": "Rakuten_Fusion_V2"}
        )

        _TUNING_PARENT_RUN_ID = parent_run.info.run_id

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

            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_val_f1_macro", study.best_value)
            mlflow.log_metric("n_trials_completed", len(trial_completed))
            mlflow.log_artifact(
                str(models_dir / "best_hyperparams.joblib"),
                artifact_path="best_params",
            )
        else:
            print("No trials completed.")
