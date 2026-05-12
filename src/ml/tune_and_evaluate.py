"""
Tuning and evaluation script
- Builds an imblearn pipeline with StandardScaler, SMOTE, and a classifier
- Runs RandomizedSearchCV internally and nested CV externally
- Saves best pipeline and metrics
"""

import json
import joblib
import numpy as np
import pandas as pd
from time import time

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
    from scipy.stats import randint
    from src.data.load_data import load_cleaned_data
    from src.ml.preprocess import split_features_target
    from src.utils.paths import MODELS_DIR
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
    from scipy.stats import randint
    from src.data.load_data import load_cleaned_data
    from src.ml.preprocess import split_features_target
    from src.utils.paths import MODELS_DIR

RANDOM_SEARCH_ITERS = 20
RANDOM_STATE = 42

def run_tuning(save_dir: str = None):
    df = load_cleaned_data()
    X, y = split_features_target(df)

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    param_dist = {
        "clf__n_estimators": randint(50, 500),
        "clf__max_depth": randint(3, 30),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ["sqrt", "log2", None],
    }

    inner_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=RANDOM_SEARCH_ITERS,
        scoring="roc_auc",
        cv=inner_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    # Nested CV to estimate generalization (external CV)
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)

    print("[tune] Starting nested CV (this may take a while)...")
    start = time()
    nested_scores = cross_val_score(random_search, X, y, cv=outer_cv, scoring="roc_auc", n_jobs=-1)
    elapsed = time() - start
    print(f"[tune] Nested CV ROC-AUC: {nested_scores.mean():.4f} ± {nested_scores.std():.4f} (took {elapsed:.1f}s)")

    # Fit random search on full data to get best model
    print("[tune] Fitting RandomizedSearchCV on full data to obtain best estimator...")
    random_search.fit(X, y)
    best = random_search.best_estimator_
    best_params = random_search.best_params_
    print(f"[tune] Best params: {best_params}")

    # Save pipeline and metrics
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = MODELS_DIR / "best_pipeline.pkl"
    joblib.dump(best, pipeline_path)

    metrics = {
        "nested_cv_mean_roc_auc": float(nested_scores.mean()),
        "nested_cv_std_roc_auc": float(nested_scores.std()),
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in best_params.items()},
    }

    metrics_path = MODELS_DIR / "tuning_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[tune] Saved best pipeline to: {pipeline_path}")
    print(f"[tune] Saved tuning metrics to: {metrics_path}")
    return best, metrics

if __name__ == "__main__":
    run_tuning()
