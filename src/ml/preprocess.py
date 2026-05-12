import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
    from src.utils.constants import FEATURE_NAMES, TARGET_COLUMN, SEX_ENCODING
    from src.utils.paths import SCALER_PATH, FEATURE_NAMES_PATH
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from sklearn.preprocessing import StandardScaler
    from src.utils.constants import FEATURE_NAMES, TARGET_COLUMN, SEX_ENCODING
    from src.utils.paths import SCALER_PATH, FEATURE_NAMES_PATH

def split_features_target(df: pd.DataFrame):
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)

def save_scaler(scaler, path=SCALER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path=SCALER_PATH) -> StandardScaler:
    return joblib.load(path)

def save_feature_names(names, path=FEATURE_NAMES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(names, path)

def load_feature_names(path=FEATURE_NAMES_PATH) -> list[str]:
    return joblib.load(path)

def preprocess_single_input(patient: dict, scaler: StandardScaler) -> np.ndarray:
    row = {}
    for feat in FEATURE_NAMES:
        val = patient[feat]
        if feat == "Sex":
            sex_value = str(val).strip().lower()
            if sex_value in {"male", "m"}:
                val = "m"
            elif sex_value in {"female", "f"}:
                val = "f"
            val = SEX_ENCODING.get(str(val).lower(), val)
        row[feat] = float(val)
    df = pd.DataFrame([row], columns=FEATURE_NAMES)
    return scaler.transform(df)