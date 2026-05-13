"""
Extended debug to test different input scenarios and identify GUI-side issues
"""
import json
from src.ml.predict import predict_liver_risk
from src.utils.constants import FEATURE_NAMES

print("=" * 80)
print("GUI INPUT FORMAT TEST - Multiple Scenarios")
print("=" * 80)

test_cases = [
    {
        "name": "Healthy male (normal values)",
        "data": {
            "Age": 30,
            "Sex": "m",
            "ALB": 45.0,
            "ALP": 70.0,
            "ALT": 25.0,
            "AST": 25.0,
            "BIL": 0.8,
            "CHE": 8.0,
            "CHOL": 4.5,
            "CREA": 0.9,
            "GGT": 30.0,
            "PROT": 72.0,
        }
    },
    {
        "name": "Healthy female (normal values, Sex='f')",
        "data": {
            "Age": 40,
            "Sex": "f",
            "ALB": 43.0,
            "ALP": 65.0,
            "ALT": 22.0,
            "AST": 28.0,
            "BIL": 0.7,
            "CHE": 7.5,
            "CHOL": 4.2,
            "CREA": 0.85,
            "GGT": 25.0,
            "PROT": 71.0,
        }
    },
    {
        "name": "Elevated enzymes (should be possible risk)",
        "data": {
            "Age": 50,
            "Sex": "male",
            "ALB": 35.0,
            "ALP": 150.0,
            "ALT": 120.0,
            "AST": 130.0,
            "BIL": 1.5,
            "CHE": 3.0,
            "CHOL": 3.0,
            "CREA": 1.2,
            "GGT": 100.0,
            "PROT": 60.0,
        }
    },
    {
        "name": "Mixed with string types (could be issue)",
        "data": {
            "Age": "30",  # string instead of int
            "Sex": "M",   # uppercase
            "ALB": "45.0",
            "ALP": "70.0",
            "ALT": "25.0",
            "AST": "25.0",
            "BIL": "0.8",
            "CHE": "8.0",
            "CHOL": "4.5",
            "CREA": "0.9",
            "GGT": "30.0",
            "PROT": "72.0",
        }
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n[Test {i}] {test_case['name']}")
    print("-" * 80)
    try:
        result = predict_liver_risk(test_case["data"])
        print(f"  Prediction: {result['prediction_label']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Key markers: {result['key_markers']}")
        print(f"  Recommendation: {result['recommendation']}")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY: If Test 1 & 2 show 'Low Risk' but Test 3 shows 'Possible Risk',")
print("then the model is working correctly. Check what input GUI is actually sending.")
print("=" * 80)
