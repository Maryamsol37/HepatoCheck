#!/usr/bin/env python
from src.ml.predict import predict_liver_risk

high_risk = {
    "Age": 58, "Sex": "m",
    "ALB": 28.0, "ALP": 240.0, "ALT": 180.0, "AST": 165.0, "BIL": 4.2,
    "CHE": 3.0, "CHOL": 7.5, "CREA": 1.8, "GGT": 320.0, "PROT": 58.0
}

low_risk = {
    "Age": 30, "Sex": "m",
    "ALB": 45.0, "ALP": 70.0, "ALT": 25.0, "AST": 25.0, "BIL": 0.8,
    "CHE": 8.0, "CHOL": 4.5, "CREA": 0.9, "GGT": 30.0, "PROT": 72.0
}

print("=" * 60)
print("HIGH RISK PATIENT")
print("=" * 60)
result_high = predict_liver_risk(high_risk)
print(f"Prediction: {result_high['prediction_label']}")
print(f"Confidence: {result_high['confidence']:.1%}")
print(f"Key Markers: {', '.join(result_high['key_markers'][:4])}")

print("\n" + "=" * 60)
print("LOW RISK PATIENT")
print("=" * 60)
result_low = predict_liver_risk(low_risk)
print(f"Prediction: {result_low['prediction_label']}")
print(f"Confidence: {result_low['confidence']:.1%}")
print(f"Key Markers: {result_low['key_markers']}")
