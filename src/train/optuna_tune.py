"""
Optuna Hyperparameter Tuning Script
Runs hyperparameter optimization for YOLO11 with MLflow tracking.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import optuna
import yaml
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from ultralytics import YOLO

from src.common.repro import log_reproducibility_info


def load_config(config_path: Path) -> dict[str, Any]:
    """Load Optuna configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def sample_hyperparams(trial: optuna.Trial, search_space: dict) -> dict[str, Any]:
    """
    Sample hyperparameters from search space.
    
    Args:
        trial: Optuna trial
        search_space: Search space configuration
        
    Returns:
        Dictionary of sampled hyperparameters
    """
    params = {}
    
    for name, spec in search_space.items():
        param_type = spec.get("type", "float")
        
        if param_type == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False)
            )
        elif param_type == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"]
            )
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(
                name,
                spec["choices"]
            )
    
    return params


def create_objective(
    config: dict[str, Any],
    data_yaml: Path,
    config_path: Path | None = None
):
    """
    Create Optuna objective function.
    
    Args:
        config: Optuna configuration
        data_yaml: Path to dataset YAML
        config_path: Path to config file for logging
        
    Returns:
        Objective function for Optuna
    """
    model_config = config.get("model", {})
    search_space = config.get("search_space", {})
    output_config = config.get("output", {})
    mlflow_config = config.get("mlflow", {})
    repro_config = config.get("reproducibility", {})
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function to minimize/maximize."""
        
        # Sample hyperparameters
        params = sample_hyperparams(trial, search_space)
        
        # Start nested MLflow run
        run_name = f"trial-{trial.number}"
        
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log trial params to MLflow
            mlflow.log_params(params)
            mlflow.log_param("trial_number", trial.number)
            
            # Log reproducibility info
            log_reproducibility_info(config_path)
            
            # Initialize model
            model_name = model_config.get("name", "yolo11n.pt")
            model = YOLO(model_name)
            
            # Prepare training args
            train_args = {
                "data": str(data_yaml),
                "epochs": params.get("epochs", 3),
                "imgsz": params.get("imgsz", 640),
                "batch": params.get("batch", 16),
                "lr0": params.get("lr0", 0.01),
                "seed": repro_config.get("seed", 42),
                "project": output_config.get("project", "/app/data/runs"),
                "name": f"{output_config.get('name', 'optuna')}_trial_{trial.number}",
                "exist_ok": True,
                "verbose": False,
            }
            
            # Add augmentation params if sampled
            for aug_param in ["mosaic", "mixup", "hsv_h", "hsv_s"]:
                if aug_param in params:
                    train_args[aug_param] = params[aug_param]
            
            print(f"\n{'='*60}")
            print(f"Trial {trial.number}: {params}")
            print(f"{'='*60}\n")
            
            # Train model
            try:
                results = model.train(**train_args)
                
                # Get mAP50 as objective value
                metrics = results.results_dict if hasattr(results, "results_dict") else {}
                mAP50 = metrics.get("metrics/mAP50(B)", 0.0)
                mAP50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
                
                # Log metrics
                mlflow.log_metric("mAP50", mAP50)
                mlflow.log_metric("mAP50-95", mAP50_95)
                
                # Report for pruning
                trial.report(mAP50, step=train_args["epochs"])
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                print(f"Trial {trial.number} - mAP50: {mAP50:.4f}")
                
                return mAP50
                
            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                return 0.0
    
    return objective


def run_optuna_study(
    config: dict[str, Any],
    data_yaml: Path,
    config_path: Path | None = None
) -> optuna.Study:
    """
    Run Optuna hyperparameter study.
    
    Args:
        config: Optuna configuration
        data_yaml: Path to dataset YAML
        config_path: Path to config file
        
    Returns:
        Completed Optuna study
    """
    optuna_config = config.get("optuna", {})
    mlflow_config = config.get("mlflow", {})
    
    # Setup MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = mlflow_config.get("experiment_name", "yolo11-object-detection")
    mlflow.set_experiment(experiment_name)
    
    # Create Optuna study
    sampler = TPESampler(seed=config.get("reproducibility", {}).get("seed", 42))
    
    pruner_config = optuna_config.get("pruner", {})
    pruner = MedianPruner(
        n_startup_trials=pruner_config.get("n_startup_trials", 2),
        n_warmup_steps=pruner_config.get("n_warmup_steps", 1)
    )
    
    study = optuna.create_study(
        study_name=optuna_config.get("study_name", "yolo11-hpo"),
        direction=optuna_config.get("direction", "maximize"),
        sampler=sampler,
        pruner=pruner
    )
    
    # Start parent MLflow run
    parent_run_name = mlflow_config.get(
        "parent_run_name", 
        f"optuna-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        # Log study info
        mlflow.log_param("study_name", optuna_config.get("study_name"))
        mlflow.log_param("n_trials", optuna_config.get("n_trials", 7))
        mlflow.log_param("direction", optuna_config.get("direction", "maximize"))
        log_reproducibility_info(config_path)
        
        # Create objective
        objective = create_objective(config, data_yaml, config_path)
        
        # Run optimization
        n_trials = optuna_config.get("n_trials", 7)
        print(f"\n{'='*60}")
        print(f"Starting Optuna study with {n_trials} trials")
        print(f"{'='*60}\n")
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Log best results
        mlflow.log_metric("best_mAP50", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        
        print(f"\n{'='*60}")
        print("Optuna Study Complete!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best mAP50: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        print(f"Parent Run ID: {parent_run.info.run_id}")
        print(f"{'='*60}\n")
    
    return study


def train_best_model(
    study: optuna.Study,
    config: dict[str, Any],
    data_yaml: Path,
    config_path: Path | None = None
) -> tuple[YOLO, str]:
    """
    Train final model with best hyperparameters.
    
    Args:
        study: Completed Optuna study
        config: Configuration
        data_yaml: Path to dataset YAML
        config_path: Path to config file
        
    Returns:
        Tuple of (trained model, MLflow run ID)
    """
    model_config = config.get("model", {})
    mlflow_config = config.get("mlflow", {})
    repro_config = config.get("reproducibility", {})
    
    best_params = study.best_params
    
    # Setup MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = mlflow_config.get("experiment_name", "yolo11-object-detection")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="optuna-best-final") as run:
        mlflow.log_params(best_params)
        mlflow.log_param("source", "optuna_best")
        log_reproducibility_info(config_path)
        
        # Initialize model
        model = YOLO(model_config.get("name", "yolo11n.pt"))
        
        # Train with best params (more epochs for final model)
        train_args = {
            "data": str(data_yaml),
            "epochs": max(best_params.get("epochs", 3), 5),  # At least 5 epochs
            "imgsz": best_params.get("imgsz", 640),
            "batch": best_params.get("batch", 16),
            "lr0": best_params.get("lr0", 0.01),
            "seed": repro_config.get("seed", 42),
            "project": "/app/data/runs",
            "name": "optuna_best_final",
            "exist_ok": True,
            "verbose": True,
        }
        
        # Add augmentation params
        for aug_param in ["mosaic", "mixup", "hsv_h", "hsv_s"]:
            if aug_param in best_params:
                train_args[aug_param] = best_params[aug_param]
        
        results = model.train(**train_args)
        
        # Log metrics
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
            mlflow.log_metric("mAP50", metrics.get("metrics/mAP50(B)", 0.0))
            mlflow.log_metric("mAP50-95", metrics.get("metrics/mAP50-95(B)", 0.0))
        
        # Log artifacts
        run_dir = Path(train_args["project"]) / train_args["name"]
        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), "weights")
        
        return model, run.info.run_id


def main():
    """Main entry point for Optuna tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/app/configs/optuna.yaml"),
        help="Path to Optuna configuration"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/app/data/coco128.yaml"),
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--train-best",
        action="store_true",
        help="Train final model with best hyperparameters"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run study
    study = run_optuna_study(config, args.data, args.config)
    
    # Optionally train best model
    if args.train_best:
        model, run_id = train_best_model(study, config, args.data, args.config)
        print(f"Best model trained. Run ID: {run_id}")


if __name__ == "__main__":
    main()
