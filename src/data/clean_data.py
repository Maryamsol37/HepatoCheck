import pandas as pd

try:
    from src.utils.constants import BINARY_LABEL_MAP, SEX_ENCODING, FEATURE_NAMES, TARGET_COLUMN
    from src.utils.paths import RAW_DATA_PATH, CLEANED_DATA_PATH
    from src.data.load_data import load_raw_data
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.utils.constants import BINARY_LABEL_MAP, SEX_ENCODING, FEATURE_NAMES, TARGET_COLUMN
    from src.utils.paths import RAW_DATA_PATH, CLEANED_DATA_PATH
    from src.data.load_data import load_raw_data

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Task 3 — Binary label mapping
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(BINARY_LABEL_MAP)
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Encode Sex: m→0, f→1
    df["Sex"] = df["Sex"].str.lower().map(SEX_ENCODING)

    # Fill missing numerical values with column median
    numerical_cols = [c for c in FEATURE_NAMES if c != "Sex"]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    df = df[FEATURE_NAMES + [TARGET_COLUMN]]
    print(f"[clean_data] Class distribution:\n{df[TARGET_COLUMN].value_counts()}")
    return df

def save_cleaned_data(df: pd.DataFrame, path=CLEANED_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[clean_data] Cleaned data saved to: {path}")

if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    save_cleaned_data(cleaned)
    print(cleaned.head())