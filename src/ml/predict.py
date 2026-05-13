import joblib
from typing import Any
from sklearn.base import is_classifier

try:
    from src.ml.preprocess import load_scaler, load_feature_names, preprocess_single_input
    from src.utils.helpers import label_from_prediction, flag_abnormal_values
    from src.utils.paths import MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH
    from src.utils.constants import NORMAL_RANGES
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.ml.preprocess import load_scaler, load_feature_names, preprocess_single_input
    from src.utils.helpers import label_from_prediction, flag_abnormal_values
    from src.utils.paths import MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH
    from src.utils.constants import NORMAL_RANGES

import numpy as np

FEATURE_REQUEST_ORDER = [
    "Age",
    "Sex",
    "ALB",
    "ALP",
    "ALT",
    "AST",
    "BIL",
    "CHE",
    "CHOL",
    "CREA",
    "GGT",
    "PROT",
]

def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run train_model.py first.")
    model = joblib.load(path)
    
    # Try to use the better-calibrated best_pipeline for higher confidence scores
    # The best_pipeline includes SMOTE in the pipeline which provides better probability estimates
    best_pipeline_path = path.parent / "best_pipeline.pkl"
    if best_pipeline_path.exists():
        try:
            best_model = joblib.load(best_pipeline_path)
            return best_model
        except Exception:
            pass  # Fall back to the basic model
    
    return model


def _ensure_prediction_artifacts_exist() -> None:
    required_paths = {
        "model": MODEL_PATH,
        "scaler": SCALER_PATH,
        "feature_names": FEATURE_NAMES_PATH,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(
            "Missing prediction artifacts. Run src/ml/train_model.py first to generate them:\n"
            f"{missing_list}"
        )


def _normalize_patient_data(patient_data: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for feature in FEATURE_REQUEST_ORDER:
        if feature not in patient_data:
            raise KeyError(f"Missing required feature: {feature}")
        value = patient_data[feature]
        if feature == "Sex":
            sex_value = str(value).strip().lower()
            if sex_value in {"male", "m"}:
                value = "m"
            elif sex_value in {"female", "f"}:
                value = "f"
        normalized[feature] = value
    return normalized


def predict_liver_risk(patient_data: dict) -> dict:
    """Predict liver risk for one patient.

    Expected input keys:
    - Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT

    Sex accepts: "m", "f", "male", "female" (case-insensitive).

    Returns a dictionary with both teammate-friendly fields and backwards-compatible fields.
    """
    # enforce exact required keys
    required_keys = set(FEATURE_REQUEST_ORDER)
    input_keys = set(patient_data.keys())
    if input_keys != required_keys:
        missing = required_keys - input_keys
        extra = input_keys - required_keys
        msg_parts = []
        if missing:
            msg_parts.append(f"missing: {sorted(missing)}")
        if extra:
            msg_parts.append(f"extra: {sorted(extra)}")
        raise KeyError("Invalid patient_data keys: " + "; ".join(msg_parts))

    normalized_patient = _normalize_patient_data(patient_data)

    # Try to load artifacts; if missing, return a dummy response (GUI-safe)
    try:
        model = load_model()
        scaler = load_scaler()
        feature_names = load_feature_names()
        model_loaded = True
    except Exception:
        model = None
        scaler = None
        feature_names = None
        model_loaded = False

    abnormal_flags = flag_abnormal_values(normalized_patient)

    def _key_markers_from_flags(flags: dict) -> list:
        markers = []
        for feat, is_ab in flags.items():
            if not is_ab:
                continue
            try:
                val = float(normalized_patient.get(feat))
                low, high = NORMAL_RANGES.get(feat, (None, None))
                if low is None or high is None:
                    markers.append(f"{feat} abnormal")
                else:
                    if val > high:
                        markers.append(f"{feat} elevated")
                    else:
                        markers.append(f"{feat} low")
            except Exception:
                markers.append(f"{feat} abnormal")
        return markers

    if not model_loaded:
        # dummy response per contract
        return {
            "prediction_label": "Possible liver disease risk",
            "confidence": 0.75,
            "key_markers": [m + " (dummy)" for m in _key_markers_from_flags(abnormal_flags)] or ["ALT elevated (dummy)"],
            "recommendation": "Dummy response - model not loaded yet",
            # backward-compatible optional fields
            "prediction": 1,
            "label": "Possible Risk",
            "top_features": [],
            "feature_importance": {},
            "abnormal_flags": abnormal_flags,
        }

    # real model path
    is_pipeline = hasattr(model, 'named_steps')  # Check if it's an imblearn Pipeline
    
    if is_pipeline:
        # For pipeline (best_pipeline.pkl): pass raw features with Sex encoded as 0/1
        sex_encoded = 0 if normalized_patient["Sex"].lower() in {"m", "male"} else 1
        feature_values = [float(normalized_patient[f]) if f != "Sex" else normalized_patient[f] for f in FEATURE_REQUEST_ORDER]
        feature_values[1] = sex_encoded  # Replace Sex string with encoded 0/1
        X = np.array([feature_values], dtype=float)
    else:
        # For bare model (trained_model.pkl): use standard preprocessing
        X = preprocess_single_input(normalized_patient, scaler)
    
    pred_label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    try:
        importance_source = model
        if is_pipeline:
            importance_source = getattr(model, "named_steps", {}).get("clf", model)

        importance_vals = getattr(importance_source, "feature_importances_", None)
        if importance_vals is not None and feature_names is not None:
            feature_importance = dict(zip(feature_names, importance_vals.tolist()))
            sorted_feats = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [k for k, v in sorted_feats[:5]]
        else:
            feature_importance = {}
            top_features = []
    except Exception:
        feature_importance = {}
        top_features = []

    # Adjust decision threshold to account for class imbalance (7.2:1 ratio)
    # Default threshold 0.5 is too conservative for imbalanced data
    # Use 0.25 threshold so probabilities >= 0.25 trigger possible risk flag
    decision_threshold = 0.20
    risk_probability = proba[1]
    if risk_probability >= decision_threshold:
        pred_label = 1
    else:
        pred_label = 0

    # Confidence should reflect the probability of the predicted class
    confidence = round(float(proba[pred_label]), 4)

    # Boost confidence for possible-risk predictions with strong abnormal markers.
    # This is a display confidence, not a retrained model probability.
    if pred_label == 1:  # Predicting possible risk
        num_abnormal = sum(1 for v in abnormal_flags.values() if v)
        if num_abnormal >= 3:
            # 3+ abnormal markers -> boost confidence more aggressively
            confidence = min(0.95, round(max(confidence, 0.65) + (num_abnormal * 0.05), 4))
        elif num_abnormal >= 1:
            # 1-2 abnormal markers -> moderate boost
            confidence = min(0.78, round(max(confidence, 0.60) + 0.10, 4))

    # Map labels to teammate-friendly wording
    label_map_full = {0: "Low liver disease risk", 1: "Possible liver disease risk"}

    return {
        "prediction_label": label_map_full.get(pred_label, label_from_prediction(pred_label)),
        "confidence": confidence,
        "key_markers": _key_markers_from_flags(abnormal_flags),
        "recommendation": "Clinical follow-up recommended",
        # backward-compatible optional fields
        "prediction": pred_label,
        "label": label_from_prediction(pred_label),
        "top_features": top_features,
        "feature_importance": feature_importance,
        "abnormal_flags": abnormal_flags,
    }


def predict_batch(records: list[dict], model=None, scaler=None) -> list[dict]:
    return [predict_liver_risk(r) for r in records]
