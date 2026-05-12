from src.ml.predict import predict_liver_risk


def test_predict_liver_risk_contract():
    dummy_patient = {
        "Age": 57,
        "Sex": "Male",
        "ALB": 40.0,
        "ALP": 80.0,
        "ALT": 20.0,
        "AST": 25.0,
        "BIL": 0.8,
        "CHE": 8.0,
        "CHOL": 5.0,
        "CREA": 0.9,
        "GGT": 30.0,
        "PROT": 72.0,
    }

    result = predict_liver_risk(dummy_patient)

    assert isinstance(result, dict)
    assert "prediction_label" in result
    assert "confidence" in result
    assert "key_markers" in result
    assert "recommendation" in result
    assert result["prediction_label"] in {"Low Risk", "Possible Risk"}
    assert 0.0 <= float(result["confidence"]) <= 1.0
