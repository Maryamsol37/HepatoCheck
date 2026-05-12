import pandas as pd

try:
    from src.utils.paths import RAW_DATA_PATH, CLEANED_DATA_PATH
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import RAW_DATA_PATH, CLEANED_DATA_PATH

def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load_data] Loaded raw data from: {path}")
    return df

def load_cleaned_data(path=CLEANED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {path}. Run clean_data.py first."
        )
    df = pd.read_csv(path)
    print(f"[load_data] Loaded cleaned data from: {path}")
    return df

if __name__ == '__main__':
    try:
        load_raw_data()
    except Exception as e:
        print(f'Error: {e}')
