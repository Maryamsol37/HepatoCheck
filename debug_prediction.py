"""
Debug script to trace prediction pipeline step-by-step
"""
import json
import joblib
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ml.preprocess import load_scaler, load_feature_names, preprocess_single_input
from src.ml.predict import predict_liver_risk
from src.utils.paths import MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH
from src.utils.constants import FEATURE_NAMES

# Test case: "healthy" patient with normal lab values
test_patient = {
    "Age": 30,
    "Sex": "m",
    "ALB": 45.0,      # normal (35-52)
    "ALP": 70.0,      # normal (30-120)
    "ALT": 25.0,      # normal (7-56)
    "AST": 25.0,      # normal (10-40)
    "BIL": 0.8,       # normal (0.1-1.2)
    "CHE": 8.0,       # normal (4-13)
    "CHOL": 4.5,      # normal (3-6.2)
    "CREA": 0.9,      # normal (0.6-1.3)
    "GGT": 30.0,      # normal (8-61)
    "PROT": 72.0,     # normal (64-83)
}

print("=" * 80)
print("DEBUG: Prediction Pipeline Trace")
print("=" * 80)

# Step 1: Show input
print("\n[1] INPUT PATIENT DATA:")
print(json.dumps(test_patient, indent=2))

# Step 2: Load artifacts
print("\n[2] LOADING ARTIFACTS:")
try:
    model = joblib.load(MODEL_PATH)
    scaler = load_scaler()
    feature_names = load_feature_names()
    print(f"  ✓ Model loaded from {MODEL_PATH}")
    print(f"  ✓ Scaler loaded from {SCALER_PATH}")
    print(f"  ✓ Feature names loaded from {FEATURE_NAMES_PATH}")
    print(f"  Feature order: {feature_names}")
except Exception as e:
    print(f"  ✗ Error loading artifacts: {e}")
    sys.exit(1)

# Step 3: Preprocess
print("\n[3] PREPROCESSING:")
try:
    X_scaled = preprocess_single_input(test_patient, scaler)
    print(f"  Scaled input shape: {X_scaled.shape}")
    print(f"  Scaled input values:\n{X_scaled}")
except Exception as e:
    print(f"  ✗ Error during preprocessing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Model prediction
print("\n[4] MODEL PREDICTION:")
try:
    pred_label = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    print(f"  Predicted label: {pred_label}")
    print(f"  Class probabilities: {proba}")
    print(f"    - P(Low Risk) = {proba[0]:.4f}")
    print(f"    - P(Possible Risk) = {proba[1]:.4f}")
    print(f"  Max confidence: {max(proba):.4f}")
except Exception as e:
    print(f"  ✗ Error during model prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Full prediction contract
print("\n[5] FULL PREDICTION RESULT:")
try:
    result = predict_liver_risk(test_patient)
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"  ✗ Error in predict_liver_risk: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Feature importance check
print("\n[6] FEATURE IMPORTANCE (if available):")
try:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = (-importances).argsort()[:5]
        print("  Top 5 important features:")
        for i, idx in enumerate(sorted_idx, 1):
            print(f"    {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    else:
        print("  Model does not have feature_importances_ attribute")
except Exception as e:
    print(f"  ✗ Error accessing feature importance: {e}")

print("\n" + "=" * 80)
