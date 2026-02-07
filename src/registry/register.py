"""
Model Registry Script
Registers trained models to MLflow Model Registry.
"""

import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def register_model(
    run_id: str,
    model_name: str = "detector",
    artifact_path: str = "weights/best.pt",
    description: str | None = None,
    tags: dict[str, str] | None = None
) -> tuple[str, int]:
    """
    Register a model from an MLflow run to the Model Registry.
    
    Args:
        run_id: MLflow run ID containing the model artifact
        model_name: Name for the registered model
        artifact_path: Path to model artifact within the run
        description: Optional model description
        tags: Optional tags for the model version
        
    Returns:
        Tuple of (model name, version number)
    """
    # Setup MLflow client
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    # Construct model URI
    model_uri = f"runs:/{run_id}/{artifact_path}"
    
    print(f"\n{'='*60}")
    print(f"Registering model from run: {run_id}")
    print(f"Model URI: {model_uri}")
    print(f"Model name: {model_name}")
    print(f"{'='*60}\n")
    
    # Register the model
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags=tags
    )
    
    version = int(result.version)
    print(f"Registered model '{model_name}' version {version}")
    
    # Set description if provided
    if description:
        client.update_model_version(
            name=model_name,
            version=version,
            description=description
        )
    
    return model_name, version


def transition_model_stage(
    model_name: str,
    version: int,
    stage: str = "Production"
) -> None:
    """
    Transition a model version to a new stage.
    
    Args:
        model_name: Name of the registered model
        version: Version number
        stage: Target stage (Production, Staging, Archived, None)
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    print(f"Transitioned {model_name} v{version} to stage: {stage}")


def get_latest_version(
    model_name: str,
    stage: str | None = None
) -> dict[str, Any] | None:
    """
    Get the latest version of a registered model.
    
    Args:
        model_name: Name of the registered model
        stage: Optional stage filter
        
    Returns:
        Model version info or None if not found
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    try:
        if stage:
            versions = client.get_latest_versions(model_name, stages=[stage])
        else:
            versions = client.get_latest_versions(model_name)
        
        if versions:
            latest = versions[0]
            return {
                "name": latest.name,
                "version": int(latest.version),
                "stage": latest.current_stage,
                "run_id": latest.run_id,
                "source": latest.source,
            }
    except Exception as e:
        print(f"Error getting latest version: {e}")
    
    return None


def compare_versions(
    model_name: str,
    versions: list[int] | None = None
) -> list[dict[str, Any]]:
    """
    Compare metrics across model versions.
    
    Args:
        model_name: Name of the registered model
        versions: Optional list of versions to compare (all if None)
        
    Returns:
        List of version info with metrics
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    # Get all versions
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    if versions:
        all_versions = [v for v in all_versions if int(v.version) in versions]
    
    results = []
    for mv in all_versions:
        # Get run metrics
        try:
            run = client.get_run(mv.run_id)
            metrics = run.data.metrics
        except Exception:
            metrics = {}
        
        results.append({
            "version": int(mv.version),
            "stage": mv.current_stage,
            "run_id": mv.run_id,
            "mAP50": metrics.get("mAP50", metrics.get("eval_mAP50", None)),
            "mAP50-95": metrics.get("mAP50-95", metrics.get("eval_mAP50-95", None)),
        })
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"Model Versions: {model_name}")
    print(f"{'='*70}")
    print(f"{'Version':>8} {'Stage':<12} {'mAP50':>10} {'mAP50-95':>12}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x["version"]):
        mAP50 = f"{r['mAP50']:.4f}" if r["mAP50"] else "N/A"
        mAP50_95 = f"{r['mAP50-95']:.4f}" if r["mAP50-95"] else "N/A"
        print(f"{r['version']:>8} {r['stage']:<12} {mAP50:>10} {mAP50_95:>12}")
    
    print(f"{'='*70}\n")
    
    return results


def main():
    """Main entry point for model registration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Register model to MLflow")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run ID containing the model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="detector",
        help="Name for the registered model"
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="weights/best.pt",
        help="Path to model artifact within the run"
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Model version description"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["Production", "Staging", "Archived"],
        help="Stage to transition model to after registration"
    )
    
    args = parser.parse_args()
    
    # Register model
    model_name, version = register_model(
        run_id=args.run_id,
        model_name=args.model_name,
        artifact_path=args.artifact_path,
        description=args.description
    )
    
    # Transition stage if specified
    if args.stage:
        transition_model_stage(model_name, version, args.stage)
    
    print(f"\nModel registered: {model_name} v{version}")


if __name__ == "__main__":
    main()
