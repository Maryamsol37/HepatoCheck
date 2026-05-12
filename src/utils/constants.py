BINARY_LABEL_MAP = {
    "0=Blood Donor":          0,   # Low Risk
    "0s=suspect Blood Donor": 0,   # Low Risk
    "1=Hepatitis":            1,   # Possible Risk
    "2=Fibrosis":             1,   # Possible Risk
    "3=Cirrhosis":            1,   # Possible Risk
}

SEX_ENCODING = {
    "m": 0,
    "f": 1,
}

FEATURE_NAMES = [
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

TARGET_COLUMN = "Category"
RANDOM_STATE = 42
TEST_SIZE = 0.2

PREDICTION_LABELS = {
    0: "Low Risk",
    1: "Possible Risk",
}

NORMAL_RANGES = {
    "ALB": (35.0, 52.0),
    "ALP": (30.0, 120.0),
    "ALT": (7.0, 56.0),
    "AST": (10.0, 40.0),
    "BIL": (0.1, 1.2),
    "CHE": (4.0, 13.0),
    "CHOL": (3.0, 6.2),
    "CREA": (0.6, 1.3),
    "GGT": (8.0, 61.0),
    "PROT": (64.0, 83.0),
}
