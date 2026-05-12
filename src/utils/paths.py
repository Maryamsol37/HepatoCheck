"""
Path Utilities
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_INPUTS_DIR = DATA_DIR / "sample_inputs"
RAW_DATA_PATH = RAW_DATA_DIR / "hcvdat0.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "hcv_cleaned.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "trained_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Source paths
SRC_DIR = PROJECT_ROOT / "src"
