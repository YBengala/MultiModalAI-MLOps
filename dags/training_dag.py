"""
Airflow DAG – Rakuten Training Pipeline
========================================
Step 1: Tuning     – Optuna hyperparameter search (20 trials) → best_hyperparams.joblib
Step 2: Training   – Final training run with best params → model_fusion_*.pth
Both steps log to MLflow (Rakuten_Multimodal_Tuning / Rakuten_Multimodal_Training).
Artifacts (model, best_hyperparams) are stored in MinIO via MLflow artifact store.

Schedule: triggered automatically via Airflow Dataset when `rakuten_ingestion` emits
the EMBEDDINGS_DATASET signal (s3://rakuten-datalake/embeddings/embeddings.parquet).
Training is skipped if a Production model already exists for the current data batch.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from dag_datasets import EMBEDDINGS_DATASET


def _get_latest_data_run_id() -> str | None:
    import mlflow

    from multimodal_ai.config.settings import settings

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        experiments = client.search_experiments(
            filter_string="name = 'Rakuten_Multimodal'"
        )
        if not experiments:
            return None
        runs = client.search_runs(
            experiment_ids=[experiments[0].experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            return runs[0].data.params.get("run_id")
    except Exception:
        pass
    return None


default_args = {
    "owner": "ybeng",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="rakuten_training",
    description="Hyperparameter tuning (Optuna) + final model training with MLflow tracking",
    default_args=default_args,
    schedule=[EMBEDDINGS_DATASET],
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["rakuten", "training", "mlflow", "optuna"],
)
def rakuten_training_dag():
    # STEP 1: HYPERPARAMETER TUNING
    @task()
    def hyperparameter_tuning() -> dict:
        """
        Run Optuna study via tuning.py logic.
        Returns best params dict via XCom.
        """
        import joblib
        import mlflow
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler

        import multimodal_ai.training.tuning as tuning_module
        from multimodal_ai.config.settings import settings
        from multimodal_ai.training.train import set_seed

        set_seed(42)

        data_run_id = _get_latest_data_run_id()

        storage = settings.OPTUNA_STORAGE_URI
        study = optuna.create_study(
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            study_name="Rakuten_Fusion_V2",
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=15),
            sampler=TPESampler(seed=42),
        )

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_TUNING_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="study_Rakuten_Fusion_V2") as parent_run:
            mlflow.set_tags(
                {
                    "pipeline": "rakuten_tuning",
                    "study_name": "Rakuten_Fusion_V2",
                    "data_run_id": data_run_id or "unknown",
                }
            )
            tuning_module._TUNING_PARENT_RUN_ID = parent_run.info.run_id
            tuning_module._TUNING_DATA_RUN_ID = data_run_id

            study.optimize(tuning_module.objective, n_trials=20)

            trial_completed = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if not trial_completed:
                raise RuntimeError("No Optuna trials completed successfully.")

            best_params = study.best_params
            best_value = study.best_value
            print(f"Best Accuracy: {best_value:.4f} | Best Params: {best_params}")

            models_dir = settings.PROJECT_ROOT / "models"
            models_dir.mkdir(exist_ok=True, parents=True)
            joblib.dump(best_params, models_dir / "best_hyperparams.joblib")

            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_f1_macro", best_value)
            mlflow.log_metric("n_trials_completed", len(trial_completed))
            mlflow.log_artifact(
                str(models_dir / "best_hyperparams.joblib"),
                artifact_path="best_params",
            )

        return best_params

    # STEP 2: FINAL TRAINING
    @task()
    def final_training(best_params: dict) -> None:
        """
        Train the final model using best hyperparameters from Optuna.
        Skips if a Production model already exists for the current data batch.
        Delegates to train_pipeline().
        """
        import mlflow

        from multimodal_ai.config.settings import settings
        from multimodal_ai.training.train import set_seed, train_pipeline

        set_seed(42)
        data_run_id = _get_latest_data_run_id()

        # Skip if Production model already trained on this batch
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        try:
            prod = client.get_latest_versions(
                "Rakuten_Multimodal_Fusion", stages=["Production"]
            )
            if prod:
                prod_run = client.get_run(prod[0].run_id)
                prod_data_run_id = prod_run.data.tags.get("data_run_id")
                if prod_data_run_id and prod_data_run_id == data_run_id:
                    print(
                        f"Production model already trained on {data_run_id} — skipping."
                    )
                    return
        except Exception:
            pass

        val_f1 = train_pipeline(cfg=best_params, data_run_id=data_run_id)
        print(f"Final training complete — val_f1_macro: {val_f1:.4f}")

    # ORCHESTRATION
    best = hyperparameter_tuning()
    final_training(best)


rakuten_training_dag()
