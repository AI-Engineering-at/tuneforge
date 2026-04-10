#!/usr/bin/env python3
"""
TurboQuant Manager — Enterprise Integration for TuneForge
Handles safe downloading, verification, and management of quantized models.

Usage:
    python scripts/turboquant_manager.py download
    python scripts/turboquant_manager.py verify
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Try to import huggingface_hub, provide clear error if missing
try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface_hub is not installed.")
    print("Please install requirements: pip install huggingface_hub")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("turboquant_manager")

def get_env_config():
    """Load configuration from env vars with defaults."""
    return {
        "hf_token": os.environ.get("HF_TOKEN"),
        "repo": os.environ.get("TURBOQUANT_MODEL_REPO", "bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF"),
        "file": os.environ.get("TURBOQUANT_MODEL_FILE", "mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"),
        "output_dir": os.environ.get("TURBOQUANT_MODELS_DIR", "/models")
    }

def verify_repo(repo_id: str, token: str = None) -> bool:
    """Verify if the HF repo exists and is accessible."""
    import requests
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/api/models/{repo_id}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        logger.info(f"Repository {repo_id} verified accessible.")
        return True
    elif response.status_code == 401:
        logger.error(f"Unauthorized (401) for {repo_id}. Check your HF_TOKEN.")
    elif response.status_code == 404:
        logger.error(f"Repository (404) not found: {repo_id}. Repo names change frequently.")
    else:
        logger.error(f"Unexpected status {response.status_code} for {repo_id}")
    return False

def download_model(config: dict):
    if not config["hf_token"]:
        logger.warning("HF_TOKEN not set in environment. Private or gated models will fail.")
        
    if not verify_repo(config["repo"], config["hf_token"]):
        sys.exit(1)
        
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting download of {config['file']} from {config['repo']}")
    logger.info(f"Destination: {out_dir}")
    
    try:
        path = hf_hub_download(
            repo_id=config['repo'],
            filename=config['file'],
            local_dir=str(out_dir),
            resume_download=True,
            token=config['hf_token']
        )
        size_gb = os.path.getsize(path) / 1e9
        logger.info(f"✅ Success! File ready at: {path}")
        logger.info(f"Size: {size_gb:.2f} GB")
    except HfHubHTTPError as e:
        logger.error(f"Failed to download: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant Model Manager")
    parser.add_argument("action", choices=["download", "verify"], help="Action to perform")
    
    args = parser.parse_args()
    config = get_env_config()
    
    if args.action == "verify":
        verify_repo(config["repo"], config["hf_token"])
    elif args.action == "download":
        download_model(config)
