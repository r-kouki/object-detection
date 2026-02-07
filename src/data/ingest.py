"""
Data Ingestion Script
Downloads and prepares COCO128 dataset for training.
"""

import os
import shutil
import subprocess
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def download_coco128(data_dir: Path) -> Path:
    """
    Download COCO128 dataset using Ultralytics.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Path to the dataset YAML file
    """
    print("=" * 60)
    print("Downloading COCO128 dataset...")
    print("=" * 60)
    
    # Ultralytics will download to the default location
    # We need to trigger the download by loading the dataset config
    dataset_yaml = "coco128.yaml"
    
    # This triggers the download if not present
    try:
        dataset_info = check_det_dataset(dataset_yaml)
        print(f"Dataset downloaded to: {dataset_info.get('path', 'unknown')}")
    except Exception as e:
        print(f"Using fallback download method: {e}")
        # Fallback: use a simple YOLO model to trigger download
        model = YOLO("yolo11n.pt")
        # Run a quick validation to trigger dataset download
        model.val(data=dataset_yaml, imgsz=64, batch=1, verbose=False)
    
    # Find the downloaded dataset
    possible_paths = [
        Path("/app/datasets/coco128"),  # Ultralytics default in container
        Path.home() / "datasets" / "coco128",
        Path("/app/data/coco128"),
        data_dir / "coco128",
    ]
    
    dataset_path = None
    for p in possible_paths:
        if p.exists():
            dataset_path = p
            break
    
    if dataset_path is None:
        raise RuntimeError("Could not find downloaded COCO128 dataset")
    
    # Copy to our data directory if needed
    target_path = data_dir / "coco128"
    if dataset_path != target_path:
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(dataset_path, target_path)
        print(f"Copied dataset to: {target_path}")
    
    # Create a local dataset YAML
    local_yaml = data_dir / "coco128.yaml"
    yaml_content = {
        "path": str(target_path),
        "train": "images/train2017",
        "val": "images/train2017",  # COCO128 uses same for train/val
        "names": {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
            10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
            14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
            20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
            34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
            38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
            42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
            47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
            52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
            57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
            61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
            66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
            70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
            75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
            79: "toothbrush"
        }
    }
    
    with open(local_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset YAML: {local_yaml}")
    return local_yaml


def setup_dvc(data_dir: Path) -> None:
    """
    Initialize DVC and add dataset to tracking.
    
    Args:
        data_dir: Directory containing the dataset
    """
    print("\n" + "=" * 60)
    print("Setting up DVC tracking...")
    print("=" * 60)
    
    project_root = data_dir.parent
    
    # Initialize DVC if not already done
    dvc_dir = project_root / ".dvc"
    if not dvc_dir.exists():
        subprocess.run(["dvc", "init"], cwd=project_root, check=True)
        print("Initialized DVC repository")
    
    # Configure remote (MinIO)
    remote_url = os.environ.get("DVC_REMOTE_URL", "s3://dvc")
    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    
    subprocess.run(
        ["dvc", "remote", "add", "-f", "minio", remote_url],
        cwd=project_root,
        check=True
    )
    subprocess.run(
        ["dvc", "remote", "modify", "minio", "endpointurl", endpoint],
        cwd=project_root,
        check=True
    )
    subprocess.run(
        ["dvc", "remote", "default", "minio"],
        cwd=project_root,
        check=True
    )
    print(f"Configured DVC remote: {remote_url} (endpoint: {endpoint})")
    
    # Add dataset to DVC
    dataset_path = data_dir / "coco128"
    if dataset_path.exists():
        subprocess.run(
            ["dvc", "add", str(dataset_path)],
            cwd=project_root,
            check=True
        )
        print(f"Added {dataset_path} to DVC tracking")
    
    # Push to remote
    try:
        subprocess.run(
            ["dvc", "push"],
            cwd=project_root,
            check=True
        )
        print("Pushed data to DVC remote")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not push to remote: {e}")


def main():
    """Main entry point for data ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest COCO128 dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/app/data"),
        help="Directory to store the dataset"
    )
    parser.add_argument(
        "--setup-dvc",
        action="store_true",
        help="Initialize and configure DVC"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push data to DVC remote"
    )
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    dataset_yaml = download_coco128(args.data_dir)
    
    # Optionally setup DVC
    if args.setup_dvc:
        setup_dvc(args.data_dir)
    
    print("\n" + "=" * 60)
    print("Data ingestion complete!")
    print(f"Dataset YAML: {dataset_yaml}")
    print("=" * 60)


if __name__ == "__main__":
    main()
