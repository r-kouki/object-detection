"""
Reproducibility Utilities
Logs git hash, DVC revision, and config hash for experiment tracking.
"""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow


def get_git_hash() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> str | None:
    """Get current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_tag() -> str | None:
    """Get current git tag if on a tagged commit."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_dvc_hash(data_path: Path | None = None) -> str | None:
    """
    Get DVC data version hash.
    
    Args:
        data_path: Optional specific data path (uses .dvc file if None)
    """
    try:
        if data_path:
            dvc_file = Path(str(data_path) + ".dvc")
            if dvc_file.exists():
                with open(dvc_file) as f:
                    content = f.read()
                    # Extract md5 hash from .dvc file
                    for line in content.split("\n"):
                        if "md5:" in line:
                            return line.split("md5:")[-1].strip()
        
        # Fallback: get DVC status
        result = subprocess.run(
            ["dvc", "version"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_config_hash(config_path: Path) -> str | None:
    """
    Get hash of configuration file.
    
    Args:
        config_path: Path to config file
    """
    try:
        with open(config_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except (FileNotFoundError, IOError):
        return None


def get_environment_info() -> dict[str, str]:
    """Get environment information for reproducibility."""
    return {
        "python_version": os.popen("python --version").read().strip(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "user": os.environ.get("USER", "unknown"),
    }


def log_reproducibility_info(
    config_path: Path | None = None,
    data_path: Path | None = None,
    zenml_run_id: str | None = None
) -> dict[str, Any]:
    """
    Log reproducibility information to MLflow.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to data directory
        zenml_run_id: Optional ZenML run ID
        
    Returns:
        Dictionary of logged information
    """
    repro_info = {}
    
    # Git info
    git_hash = get_git_hash()
    if git_hash:
        repro_info["git_commit"] = git_hash
        mlflow.set_tag("git_commit", git_hash)
    
    git_branch = get_git_branch()
    if git_branch:
        repro_info["git_branch"] = git_branch
        mlflow.set_tag("git_branch", git_branch)
    
    git_tag = get_git_tag()
    if git_tag:
        repro_info["git_tag"] = git_tag
        mlflow.set_tag("git_tag", git_tag)
    
    # DVC info
    dvc_hash = get_dvc_hash(data_path)
    if dvc_hash:
        repro_info["dvc_data_version"] = dvc_hash
        mlflow.set_tag("dvc_data_version", dvc_hash)
    
    # Config hash
    if config_path:
        config_hash = get_config_hash(config_path)
        if config_hash:
            repro_info["config_hash"] = config_hash
            mlflow.set_tag("config_hash", config_hash)
            mlflow.set_tag("config_file", str(config_path))
    
    # ZenML run ID
    if zenml_run_id:
        repro_info["zenml_run_id"] = zenml_run_id
        mlflow.set_tag("zenml_run_id", zenml_run_id)
    
    # Environment info
    env_info = get_environment_info()
    for key, value in env_info.items():
        mlflow.set_tag(key, value)
        repro_info[key] = value
    
    print("\n" + "-" * 40)
    print("Reproducibility Info:")
    for key, value in repro_info.items():
        print(f"  {key}: {value}")
    print("-" * 40 + "\n")
    
    return repro_info


def verify_reproducibility(
    git_commit: str,
    dvc_hash: str | None = None,
    config_hash: str | None = None
) -> dict[str, bool]:
    """
    Verify current state matches expected reproducibility info.
    
    Args:
        git_commit: Expected git commit hash
        dvc_hash: Expected DVC data hash
        config_hash: Expected config hash
        
    Returns:
        Dictionary of verification results
    """
    results = {}
    
    # Check git commit
    current_git = get_git_hash()
    results["git_commit_match"] = current_git == git_commit
    if not results["git_commit_match"]:
        print(f"Warning: Git mismatch - expected {git_commit}, got {current_git}")
    
    # Check DVC hash if provided
    if dvc_hash:
        current_dvc = get_dvc_hash()
        results["dvc_hash_match"] = current_dvc == dvc_hash
        if not results["dvc_hash_match"]:
            print(f"Warning: DVC mismatch - expected {dvc_hash}, got {current_dvc}")
    
    return results
