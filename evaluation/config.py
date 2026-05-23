# evaluation/config.py
import os
import json
from pathlib import Path

# Default number of evaluation episodes
DEFAULT_EPISODES = 5

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
RUNS_DIR = os.path.join(EXPERIMENTS_DIR, "runs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure required directories exist
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def load_historical_metrics():
    """
    Load historical benchmark metrics from JSON file.
    
    Returns:
        dict: Historical metrics for all agent configurations (baseline, memory, guarded, finetuned, hybrid)
              from Phase 3 Re-Thesis evaluation.
    
    Raises:
        FileNotFoundError: If historical_results.json does not exist.
        json.JSONDecodeError: If JSON file is malformed.
    """
    historical_file = Path(EVAL_DIR) / "historical_results.json"
    
    if not historical_file.exists():
        raise FileNotFoundError(
            f"Historical metrics file not found at {historical_file}. "
            "Please ensure evaluation/historical_results.json exists."
        )
    
    try:
        with open(historical_file, "r") as f:
            data = json.load(f)
        return data["results"]
    except KeyError:
        raise ValueError(
            "historical_results.json is missing 'results' key. "
            "Please verify file structure matches expected format."
        )
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Failed to parse historical_results.json: {e.msg}",
            e.doc,
            e.pos
        )

# Lazy-load historical metrics (only when first accessed)
_HISTORICAL_METRICS = None

def HISTORICAL_METRICS():
    """Backward-compatible accessor for historical metrics (deprecated)."""
    global _HISTORICAL_METRICS
    if _HISTORICAL_METRICS is None:
        _HISTORICAL_METRICS = load_historical_metrics()
    return _HISTORICAL_METRICS
