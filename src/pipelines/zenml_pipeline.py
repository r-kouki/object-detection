"""
ZenML Pipeline for YOLO11 Object Detection
Orchestrates the complete MLOps workflow: data → train → eval → register
"""

import os
from pathlib import Path
from typing import Any

import yaml
from zenml import pipeline, step, get_step_context
from zenml.client import Client


# =============================================================================
# Step 1: Data Ingestion
# =============================================================================
@step
def data_ingest_step(
    data_dir: str = "/app/data",
    setup_dvc: bool = True
) -> str:
    """
    Download and prepare dataset.
    
    Args:
        data_dir: Directory to store data
        setup_dvc: Whether to setup DVC tracking
        
    Returns:
        Path to dataset YAML
    """
    from src.data.ingest import download_coco128, setup_dvc as configure_dvc
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    dataset_yaml = download_coco128(data_path)
    
    # Setup DVC if requested
    if setup_dvc:
        try:
            configure_dvc(data_path)
        except Exception as e:
            print(f"Warning: DVC setup failed: {e}")
    
    return str(dataset_yaml)


# =============================================================================
# Step 2: Training
# =============================================================================
@step
def train_step(
    data_yaml: str,
    config_path: str = "/app/configs/train_baseline.yaml"
) -> dict:
    """
    Train YOLO model.
    
    Args:
        data_yaml: Path to dataset YAML
        config_path: Path to training config
        
    Returns:
        Training results dict
    """
    from src.train.train import load_config, train_yolo
    
    config = load_config(Path(config_path))
    
    # Get ZenML run ID for logging
    try:
        context = get_step_context()
        zenml_run_id = str(context.pipeline_run.id)
    except Exception:
        zenml_run_id = None
    
    model, metrics, mlflow_run_id = train_yolo(
        config=config,
        data_yaml=Path(data_yaml),
        config_path=Path(config_path)
    )
    
    # Return results with paths and IDs
    output_config = config.get("output", {})
    run_dir = Path(output_config.get("project", "/app/data/runs")) / output_config.get("name", "train")
    
    return {
        "mlflow_run_id": mlflow_run_id,
        "zenml_run_id": zenml_run_id,
        "weights_path": str(run_dir / "weights" / "best.pt"),
        "metrics": {
            "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
        },
        "config_path": config_path,
    }


# =============================================================================
# Step 3: Evaluation
# =============================================================================
@step
def eval_step(
    train_output: dict,
    data_yaml: str
) -> dict:
    """
    Evaluate trained model.
    
    Args:
        train_output: Output from training step
        data_yaml: Path to dataset YAML
        
    Returns:
        Evaluation metrics
    """
    from src.train.eval import evaluate_model
    
    weights_path = train_output.get("weights_path")
    config_path = train_output.get("config_path")
    mlflow_run_id = train_output.get("mlflow_run_id")
    
    # Load config if available
    config = None
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Run evaluation
    metrics = evaluate_model(
        model_path=Path(weights_path),
        data_yaml=Path(data_yaml),
        config=config,
        run_id=mlflow_run_id
    )
    
    return {
        "mlflow_run_id": mlflow_run_id,
        "metrics": metrics,
        "weights_path": weights_path,
    }


# =============================================================================
# Step 4: Model Registration
# =============================================================================
@step
def register_step(
    eval_output: dict,
    model_name: str = "detector",
    description: str = None,
    stage: str = None
) -> dict:
    """
    Register model to MLflow Model Registry.
    
    Args:
        eval_output: Output from evaluation step
        model_name: Name for registered model
        description: Optional description
        stage: Optional stage to transition to
        
    Returns:
        Registration info
    """
    from src.registry.register import register_model, transition_model_stage
    
    mlflow_run_id = eval_output.get("mlflow_run_id")
    metrics = eval_output.get("metrics", {})
    
    # Create description from metrics if not provided
    if not description:
        description = (
            f"YOLO11 detector - "
            f"mAP50: {metrics.get('mAP50', 'N/A'):.4f}, "
            f"mAP50-95: {metrics.get('mAP50-95', 'N/A'):.4f}"
        )
    
    # Register model
    registered_name, version = register_model(
        run_id=mlflow_run_id,
        model_name=model_name,
        artifact_path="weights/best.pt",
        description=description
    )
    
    # Transition stage if specified
    if stage:
        transition_model_stage(registered_name, version, stage)
    
    return {
        "model_name": registered_name,
        "version": version,
        "stage": stage or "None",
        "mlflow_run_id": mlflow_run_id,
    }


# =============================================================================
# Pipeline Definition
# =============================================================================
@pipeline(name="yolo11_object_detection")
def yolo_pipeline(
    data_dir: str = "/app/data",
    config_path: str = "/app/configs/train_baseline.yaml",
    model_name: str = "detector",
    register_model: bool = True,
    stage: str = None
):
    """
    Complete YOLO11 training pipeline.
    
    Args:
        data_dir: Directory for data storage
        config_path: Path to training configuration
        model_name: Name for model registration
        register_model: Whether to register the model
        stage: Stage for registered model
    """
    # Step 1: Data ingestion
    data_yaml = data_ingest_step(data_dir=data_dir)
    
    # Step 2: Training
    train_output = train_step(
        data_yaml=data_yaml,
        config_path=config_path
    )
    
    # Step 3: Evaluation
    eval_output = eval_step(
        train_output=train_output,
        data_yaml=data_yaml
    )
    
    # Step 4: Registration (optional)
    if register_model:
        register_step(
            eval_output=eval_output,
            model_name=model_name,
            stage=stage
        )


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Run the ZenML pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run YOLO11 training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="/app/configs/train_baseline.yaml",
        help="Path to training configuration"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/data",
        help="Directory for data storage"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="detector",
        help="Name for registered model"
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip model registration"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["Production", "Staging"],
        help="Stage for registered model"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Running YOLO11 Object Detection Pipeline")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Data dir: {args.data_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Register: {not args.no_register}")
    print(f"Stage: {args.stage}")
    print("=" * 60 + "\n")
    
    # Initialize ZenML (using default stack)
    try:
        client = Client()
        print(f"Using ZenML stack: {client.active_stack.name}")
    except Exception as e:
        print(f"ZenML initialization note: {e}")
    
    # Run pipeline
    result = yolo_pipeline(
        data_dir=args.data_dir,
        config_path=args.config,
        model_name=args.model_name,
        register_model=not args.no_register,
        stage=args.stage
    )
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
