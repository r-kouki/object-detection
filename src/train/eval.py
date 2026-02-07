"""
Model Evaluation Script
Evaluates trained YOLO models and logs metrics to MLflow.
"""

import os
from pathlib import Path
from typing import Any

import mlflow
import yaml
from ultralytics import YOLO


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    config: dict[str, Any] | None = None,
    run_id: str | None = None
) -> dict[str, Any]:
    """
    Evaluate a trained YOLO model.
    
    Args:
        model_path: Path to model weights (.pt file)
        data_yaml: Path to dataset YAML
        config: Optional configuration dict
        run_id: Optional MLflow run ID to log to
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(str(model_path))
    
    # Get eval settings from config
    config = config or {}
    train_config = config.get("training", {})
    
    # Run validation
    results = model.val(
        data=str(data_yaml),
        imgsz=train_config.get("imgsz", 640),
        batch=train_config.get("batch", 16),
        verbose=True,
        plots=True,
        save_json=True,
    )
    
    # Extract metrics
    metrics = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
    }
    
    # Per-class metrics if available
    if hasattr(results.box, "maps"):
        for i, class_map in enumerate(results.box.maps):
            if i < len(results.names):
                metrics[f"mAP50_{results.names[i]}"] = class_map
    
    # Speed metrics
    if hasattr(results, "speed"):
        metrics["preprocess_ms"] = results.speed.get("preprocess", 0)
        metrics["inference_ms"] = results.speed.get("inference", 0)
        metrics["postprocess_ms"] = results.speed.get("postprocess", 0)
    
    # Log to MLflow if run_id provided
    if run_id:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        with mlflow.start_run(run_id=run_id):
            # Log evaluation metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"eval_{key}", value)
            
            # Log plots from validation
            save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
            if save_dir and save_dir.exists():
                for plot_file in save_dir.glob("*.png"):
                    mlflow.log_artifact(str(plot_file), "eval_plots")
                
                # Log confusion matrix
                conf_matrix = save_dir / "confusion_matrix.png"
                if conf_matrix.exists():
                    mlflow.log_artifact(str(conf_matrix), "eval_plots")
    
    print(f"\n{'='*60}")
    print("Evaluation Results:")
    print(f"  mAP50: {metrics['mAP50']:.4f}")
    print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


def compare_models(
    model_paths: list[Path],
    data_yaml: Path,
    labels: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """
    Compare multiple models.
    
    Args:
        model_paths: List of paths to model weights
        data_yaml: Path to dataset YAML
        labels: Optional labels for each model
        
    Returns:
        Dictionary mapping model label to metrics
    """
    labels = labels or [f"model_{i}" for i in range(len(model_paths))]
    
    results = {}
    for path, label in zip(model_paths, labels):
        print(f"\nEvaluating {label}...")
        results[label] = evaluate_model(path, data_yaml)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("Model Comparison:")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'mAP50':>12} {'mAP50-95':>12} {'Precision':>12} {'Recall':>12}")
    print("-" * 80)
    
    for label, metrics in results.items():
        print(
            f"{label:<20} "
            f"{metrics['mAP50']:>12.4f} "
            f"{metrics['mAP50-95']:>12.4f} "
            f"{metrics['precision']:>12.4f} "
            f"{metrics['recall']:>12.4f}"
        )
    
    print(f"{'='*80}\n")
    
    return results


def main():
    """Main entry point for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 model")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model weights (.pt file)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/app/data/coco128.yaml"),
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (for eval settings)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID to log metrics to"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and args.config.exists():
        config = load_config(args.config)
    
    # Evaluate model
    metrics = evaluate_model(
        args.model,
        args.data,
        config=config,
        run_id=args.run_id
    )
    
    print(f"Evaluation complete. mAP50: {metrics['mAP50']:.4f}")


if __name__ == "__main__":
    main()
