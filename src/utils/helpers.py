"""
Helper Functions
"""

import json

from src.utils.constants import NORMAL_RANGES, PREDICTION_LABELS

def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers."""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return default


def save_json(data: dict, path) -> None:
    """Save a dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def label_from_prediction(prediction: int) -> str:
    """Map numeric prediction to a user-friendly label."""
    return PREDICTION_LABELS.get(int(prediction), "Unknown")


def flag_abnormal_values(patient: dict) -> dict:
    """Flag values outside common clinical ranges for known lab features."""
    flags = {}
    for feature, bounds in NORMAL_RANGES.items():
        if feature not in patient:
            continue
        try:
            value = float(patient[feature])
        except (TypeError, ValueError):
            continue
        low, high = bounds
        flags[feature] = value < low or value > high
    return flags
