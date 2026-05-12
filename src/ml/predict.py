import joblib
from typing import Any

try:
    from src.ml.preprocess import load_scaler, load_feature_names, preprocess_single_input
    from src.utils.helpers import label_from_prediction, flag_abnormal_values
    from src.utils.paths import MODEL_PATH
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.ml.preprocess import load_scaler, load_feature_names, preprocess_single_input
    from src.utils.helpers import label_from_prediction, flag_abnormal_values
    from src.utils.paths import MODEL_PATH

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
    return joblib.load(path)


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


def _dummy_prediction(patient_data: dict[str, Any]) -> dict[str, Any]:
    risk_score = 0.15
    recommendation = "Collect the real model artifacts before using this in production."
    return {
        "prediction": 0,
        "prediction_label": "Low Risk",
        "confidence": round(1.0 - risk_score, 4),
        "key_markers": {"dummy": True},
        "recommendation": recommendation,
        "label": "Low Risk",
        "probability_low_risk": round(1.0 - risk_score, 4),
        "probability_possible_risk": round(risk_score, 4),
        "top_features": {},
        "feature_importance": {},
        "abnormal_flags": flag_abnormal_values(patient_data),
    }

def predict_liver_risk(
    patient_data: dict[str, Any],
    model=None,
    scaler=None,
) -> dict[str, Any]:
    """Predict liver risk for one patient.

    Expected input keys:
    - Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT

    Sex accepts: "m", "f", "male", "female" (case-insensitive).

    Returns a dictionary with both teammate-friendly fields and backwards-compatible fields.
    """
    normalized_patient = _normalize_patient_data(patient_data)

    if model is None or scaler is None:
        try:
            if model is None:
                model = load_model()
            if scaler is None:
                scaler = load_scaler()
        except FileNotFoundError:
            return _dummy_prediction(normalized_patient)

    X = preprocess_single_input(normalized_patient, scaler)
    pred_label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    feature_names = load_feature_names()
    feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
    confidence = round(float(max(proba)), 4)

    return {
        "prediction": pred_label,
        "prediction_label": label_from_prediction(pred_label),
        "confidence": confidence,
        "key_markers": top_features,
        "recommendation": "Consult a clinician for confirmation and follow-up.",
        "label": label_from_prediction(pred_label),
        "probability_low_risk": round(float(proba[0]), 4),
        "probability_possible_risk": round(float(proba[1]), 4),
        "top_features": top_features,
        "feature_importance": feature_importance,
        "abnormal_flags": flag_abnormal_values(normalized_patient),
    }

def predict_batch(records: list[dict], model=None, scaler=None) -> list[dict]:
    if model  is None: model  = load_model()
    if scaler is None: scaler = load_scaler()
    return [predict_liver_risk(r, model=model, scaler=scaler) for r in records]