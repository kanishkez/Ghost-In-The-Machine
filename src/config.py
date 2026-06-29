"""Central configuration loader."""
from pathlib import Path
import yaml
import os
import random
import numpy as np

def load_config(path: str = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Respect RUN_SEED env var for multi-seed experiments
    if "RUN_SEED" in os.environ:
        cfg["project"]["seed"] = int(os.environ["RUN_SEED"])
        
    return cfg

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

CFG = load_config()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
