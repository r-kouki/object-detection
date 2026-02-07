"""
YOLO11 Training Script
Trains a YOLO model with MLflow tracking.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import yaml
from ultralytics import YOLO

from src.common.repro import log_reproducibility_info, get_config_hash


def load_config(config_path: Path) -> dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_mlflow(config: dict[str, Any]) -> str:
    """
    Setup MLflow tracking.
    
    Args:
        config: Training configuration
        
    Returns:
        MLflow run ID
    """
    mlflow_config = config.get("mlflow", {})
    
    # Set tracking URI from env or default
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    experiment_name = mlflow_config.get("experiment_name", "yolo11-object-detection")
    mlflow.set_experiment(experiment_name)
    
    return experiment_name


def train_yolo(
    config: dict[str, Any],
    data_yaml: Path,
    config_path: Path | None = None
) -> tuple[YOLO, dict[str, Any]]:
    """
    Train YOLO model with configuration.
    
    Args:
        config: Training configuration dictionary
        data_yaml: Path to dataset YAML file
        config_path: Optional path to config file for logging
        
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    model_config = config.get("model", {})
    train_config = config.get("training", {})
    aug_config = config.get("augmentation", {})
    output_config = config.get("output", {})
    mlflow_config = config.get("mlflow", {})
    repro_config = config.get("reproducibility", {})
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Start MLflow run
    run_name = mlflow_config.get("run_name", f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n{'='*60}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Experiment: {mlflow.get_experiment(run.info.experiment_id).name}")
        print(f"{'='*60}\n")
        
        # Log reproducibility info
        log_reproducibility_info(config_path)
        
        # Log all parameters
        mlflow.log_params({
            "model": model_config.get("name", "yolo11n.pt"),
            "epochs": train_config.get("epochs", 3),
            "imgsz": train_config.get("imgsz", 640),
            "batch": train_config.get("batch", 16),
            "lr0": train_config.get("lr0", 0.01),
            "optimizer": train_config.get("optimizer", "auto"),
            "seed": repro_config.get("seed", 42),
        })
        
        # Log augmentation params
        for key, value in aug_config.items():
            mlflow.log_param(f"aug_{key}", value)
        
        # Initialize model
        model_name = model_config.get("name", "yolo11n.pt")
        model = YOLO(model_name)
        print(f"Loaded model: {model_name}")
        
        # Prepare training arguments
        train_args = {
            "data": str(data_yaml),
            "epochs": train_config.get("epochs", 3),
            "imgsz": train_config.get("imgsz", 640),
            "batch": train_config.get("batch", 16),
            "optimizer": train_config.get("optimizer", "auto"),
            "lr0": train_config.get("lr0", 0.01),
            "lrf": train_config.get("lrf", 0.01),
            "momentum": train_config.get("momentum", 0.937),
            "weight_decay": train_config.get("weight_decay", 0.0005),
            "warmup_epochs": train_config.get("warmup_epochs", 1.0),
            "seed": repro_config.get("seed", 42),
            "deterministic": repro_config.get("deterministic", True),
            "project": output_config.get("project", "/app/data/runs"),
            "name": output_config.get("name", "train"),
            "exist_ok": output_config.get("exist_ok", True),
            "verbose": True,
        }
        
        # Add augmentation args
        train_args.update(aug_config)
        
        # Train the model
        print("\nStarting training...")
        results = model.train(**train_args)
        
        # Extract metrics
        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
        
        # Log metrics to MLflow
        metric_mapping = {
            "metrics/mAP50(B)": "mAP50",
            "metrics/mAP50-95(B)": "mAP50-95",
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
        }
        
        for yolo_key, mlflow_key in metric_mapping.items():
            if yolo_key in metrics:
                mlflow.log_metric(mlflow_key, metrics[yolo_key])
        
        # Log final loss values
        if "train/box_loss" in metrics:
            mlflow.log_metric("train_box_loss", metrics["train/box_loss"])
        if "train/cls_loss" in metrics:
            mlflow.log_metric("train_cls_loss", metrics["train/cls_loss"])
        
        # Log artifacts
        run_dir = Path(train_args["project"]) / train_args["name"]
        
        # Log best weights
        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), "weights")
            
        # Log training curves
        for plot_file in ["results.png", "confusion_matrix.png", "F1_curve.png", 
                          "P_curve.png", "R_curve.png", "PR_curve.png"]:
            plot_path = run_dir / plot_file
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), "plots")
        
        # Log results CSV
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv), "metrics")
        
        # Log training args
        args_yaml = run_dir / "args.yaml"
        if args_yaml.exists():
            mlflow.log_artifact(str(args_yaml), "config")
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Run ID: {run.info.run_id}")
        print(f"{'='*60}\n")
        
        return model, metrics, run.info.run_id


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO11 model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/app/configs/train_baseline.yaml"),
        help="Path to training configuration"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/app/data/coco128.yaml"),
        help="Path to dataset YAML"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train model
    model, metrics, run_id = train_yolo(config, args.data, args.config)
    
    print(f"Model trained successfully. Run ID: {run_id}")


if __name__ == "__main__":
    main()
