"""
Analyze model decision boundary and class imbalance
"""
import joblib
import numpy as np
from src.utils.paths import MODEL_PATH, METRICS_PATH
from src.data.load_data import load_cleaned_data
from src.ml.preprocess import split_features_target, fit_scaler, apply_scaler
from src.utils.constants import FEATURE_NAMES, RANDOM_STATE
import json

print("=" * 80)
print("MODEL DECISION BOUNDARY & CLASS IMBALANCE ANALYSIS")
print("=" * 80)

# Load model
model = joblib.load(MODEL_PATH)
print(f"\n[1] Model loaded from {MODEL_PATH}")
print(f"    Model type: {type(model)}")

# Load metrics
print(f"\n[2] Metrics:")
try:
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    print(json.dumps(metrics, indent=2))
except Exception as e:
    print(f"    Error reading metrics: {e}")

# Check model configuration
print(f"\n[3] Model Configuration:")
if hasattr(model, 'n_estimators'):
    print(f"    n_estimators: {model.n_estimators}")
if hasattr(model, 'max_depth'):
    print(f"    max_depth: {model.max_depth}")
if hasattr(model, 'class_weight'):
    print(f"    class_weight: {model.class_weight}")
if hasattr(model, 'criterion'):
    print(f"    criterion: {model.criterion}")

# Check decision threshold
print(f"\n[4] Decision Boundary Analysis:")
print(f"    Model's default threshold: 0.5")
print(f"    (class 0 if proba[1] < 0.5, class 1 if proba[1] >= 0.5)")

# Load data and check class distribution
print(f"\n[5] Dataset Class Distribution:")
df = load_cleaned_data()
from src.utils.constants import TARGET_COLUMN
vc = df[TARGET_COLUMN].value_counts()
print(f"    Class 0 (Low Risk): {vc.get(0, 0)} samples")
print(f"    Class 1 (Possible Risk): {vc.get(1, 0)} samples")
print(f"    Ratio (0:1): {vc.get(0, 0)}:{vc.get(1, 0)}")
print(f"    Imbalance ratio: {vc.get(0, 0) / vc.get(1, 0):.2f}:1")

# Analyze prediction probabilities on test set
print(f"\n[6] Probability Distribution on Full Dataset:")
X, y = split_features_target(df)
scaler = fit_scaler(X)
X_scaled = apply_scaler(X, scaler)
probas = model.predict_proba(X_scaled)
print(f"    P(Risk=1) statistics:")
print(f"      Mean: {probas[:, 1].mean():.4f}")
print(f"      Median: {np.median(probas[:, 1]):.4f}")
print(f"      Std: {probas[:, 1].std():.4f}")
print(f"      Min: {probas[:, 1].min():.4f}")
print(f"      Max: {probas[:, 1].max():.4f}")
print(f"      Percentiles:")
print(f"        25th: {np.percentile(probas[:, 1], 25):.4f}")
print(f"        50th: {np.percentile(probas[:, 1], 50):.4f}")
print(f"        75th: {np.percentile(probas[:, 1], 75):.4f}")
print(f"        90th: {np.percentile(probas[:, 1], 90):.4f}")
print(f"        95th: {np.percentile(probas[:, 1], 95):.4f}")

# Count predicted classes
preds = model.predict(X_scaled)
print(f"\n[7] Predicted Class Distribution:")
print(f"    Predicted 0 (Low Risk): {(preds == 0).sum()}")
print(f"    Predicted 1 (Possible Risk): {(preds == 1).sum()}")

# Confusion on real data
print(f"\n[8] True vs Predicted:")
print(f"    True 0, Predicted 0: {((y == 0) & (preds == 0)).sum()}")
print(f"    True 0, Predicted 1: {((y == 0) & (preds == 1)).sum()}")
print(f"    True 1, Predicted 0: {((y == 1) & (preds == 0)).sum()}")
print(f"    True 1, Predicted 1: {((y == 1) & (preds == 1)).sum()}")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("If most predictions are class 0 despite abnormal inputs, the model may:")
print("1. Be overfitting to the majority class (Low Risk)")
print("2. Have a very high confidence threshold for class 1")
print("3. Need calibration or threshold adjustment")
print("=" * 80)
