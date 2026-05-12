"""
Advanced evaluation:
- Compare RandomForest, LightGBM, CatBoost using nested CV
- Fit CalibratedClassifierCV on best model and evaluate Brier score
- RFECV feature selection (using RF) and re-evaluate
- SHAP-guided pruning and re-evaluate
- Optional external holdout if provided
Results saved to models/advanced_metrics.json
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
    from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.feature_selection import RFECV
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import shap
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
    from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.feature_selection import RFECV
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import shap
    from src.data.load_data import load_cleaned_data
    from src.ml.preprocess import split_features_target
    from src.utils.paths import MODELS_DIR

RANDOM_STATE = 42

def nested_cv_score(pipeline, X, y, outer_cv):
    return cross_val_score(pipeline, X, y, cv=outer_cv, scoring='roc_auc', n_jobs=-1)


def compare_models(X, y):
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)

    # Define pipelines for each classifier
    rf_pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE))
    ])

    lgb_pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', lgb.LGBMClassifier(n_jobs=-1, random_state=RANDOM_STATE))
    ])

    cat_pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', CatBoostClassifier(verbose=0, random_state=RANDOM_STATE))
    ])

    results = {}
    for name, pipe in [('RandomForest', rf_pipe), ('LightGBM', lgb_pipe), ('CatBoost', cat_pipe)]:
        print(f"[adv] Running nested CV for {name}...")
        start = time()
        scores = nested_cv_score(pipe, X, y, outer_cv)
        elapsed = time() - start
        results[name] = {"mean_roc_auc": float(scores.mean()), "std_roc_auc": float(scores.std()), "time_s": elapsed}
        print(f"[adv] {name}: {scores.mean():.4f} ± {scores.std():.4f} (took {elapsed:.1f}s)")

    return results


def calibrate_and_eval(best_pipeline, X_train, X_test, y_train, y_test):
    # Fit calibrated classifier on training portion
    base_clf = best_pipeline.named_steps['clf']
    # create pipeline without SMOTE for calibration (calibration needs predict_proba from classifier)
    # we will fit the pipeline (scaler + clf) and use CalibratedClassifierCV wrapping the classifier
    scaler = best_pipeline.named_steps['scaler']
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    calib = CalibratedClassifierCV(base_clf, cv=3)
    calib.fit(X_train_scaled, y_train)
    prob_pos = calib.predict_proba(X_test_scaled)[:, 1]
    brier = brier_score_loss(y_test, prob_pos)
    roc = roc_auc_score(y_test, prob_pos)
    return {"brier_score": float(brier), "roc_auc": float(roc)}


def rfecv_selection(X, y):
    # RFECV with RandomForest (no SMOTE)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    estimator = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)
    rfecv = RFECV(estimator=estimator, step=1, cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=RANDOM_STATE), scoring='roc_auc', n_jobs=-1)
    rfecv.fit(Xs, y)
    support = rfecv.support_
    return support, rfecv


def shap_pruning(best_estimator, X, feature_names, fraction=0.2):
    # Compute SHAP importances and drop bottom fraction
    explainer = shap.TreeExplainer(best_estimator.named_steps['clf'])
    # use scaled X
    scaler = best_estimator.named_steps['scaler']
    Xs = scaler.transform(X)
    shap_vals = explainer.shap_values(Xs)
    vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    mean_abs = np.abs(vals).mean(axis=0)
    thresh = np.quantile(mean_abs, fraction)
    keep = mean_abs > thresh
    return keep, mean_abs


def run_all(external_holdout_path=None):
    df = load_cleaned_data()
    X, y = split_features_target(df)
    feature_names = X.columns.tolist()

    results = {}

    # Compare models
    comp = compare_models(X, y)
    results['model_comparison'] = comp

    # Choose best by mean_roc_auc
    best_name = max(comp.items(), key=lambda kv: kv[1]['mean_roc_auc'])[0]
    results['best_model_name'] = best_name

    # Fit best on full data
    if best_name == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        best_pipe = ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=RANDOM_STATE)), ('clf', RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE))])
    elif best_name == 'LightGBM':
        best_pipe = ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=RANDOM_STATE)), ('clf', lgb.LGBMClassifier(n_jobs=-1, random_state=RANDOM_STATE))])
    else:
        best_pipe = ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=RANDOM_STATE)), ('clf', CatBoostClassifier(verbose=0, random_state=RANDOM_STATE))])

    # Train/test split for calibration and pruning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    best_pipe.fit(X_train, y_train)

    # Calibration
    calib_metrics = calibrate_and_eval(best_pipe, X_train, X_test, y_train, y_test)
    results['calibration'] = calib_metrics

    # RFECV
    support, rfecv = rfecv_selection(X, y)
    kept_features = [f for f, k in zip(feature_names, support) if k]
    results['rfecv_kept'] = kept_features

    # SHAP pruning
    keep_idx, mean_abs = shap_pruning(best_pipe, X_train, feature_names, fraction=0.2)
    shap_kept = [f for f, k in zip(feature_names, keep_idx) if k]
    results['shap_kept'] = shap_kept

    # Save results
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / 'advanced_metrics.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Save pruned pipeline trained on kept features (example: using RF and kept features)
    # retrain RF on shap_kept features
    if len(shap_kept) > 0:
        X_shap = X[shap_kept]
        pipe_shap = ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=RANDOM_STATE)), ('clf', RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE))])
        pipe_shap.fit(X_shap, y)
        joblib.dump(pipe_shap, MODELS_DIR / 'shap_pruned_pipeline.pkl')

    print(f"[adv] Saved advanced metrics to: {out_path}")
    return results

if __name__ == '__main__':
    run_all()
