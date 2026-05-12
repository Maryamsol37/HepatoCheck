import joblib
import pandas as pd

try:
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from src.data.load_data import load_cleaned_data
    from src.ml.preprocess import split_features_target, fit_scaler, apply_scaler, save_scaler, save_feature_names
    from src.ml.evaluate import compute_metrics, plot_confusion_matrix, plot_feature_importance
    from src.ml.explain import get_shap_explainer, plot_shap_summary
    from src.utils.constants import FEATURE_NAMES, RANDOM_STATE, TEST_SIZE
    from src.utils.helpers import save_json
    from src.utils.paths import MODEL_PATH, METRICS_PATH
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from src.data.load_data import load_cleaned_data
    from src.ml.preprocess import split_features_target, fit_scaler, apply_scaler, save_scaler, save_feature_names
    from src.ml.evaluate import compute_metrics, plot_confusion_matrix, plot_feature_importance
    from src.ml.explain import get_shap_explainer, plot_shap_summary
    from src.utils.constants import FEATURE_NAMES, RANDOM_STATE, TEST_SIZE
    from src.utils.helpers import save_json
    from src.utils.paths import MODEL_PATH, METRICS_PATH

def train(df: pd.DataFrame | None = None):
    if df is None:
        df = load_cleaned_data()

    X, y = split_features_target(df)

    # Task 4 — train/test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, scaler)
    X_test_scaled  = apply_scaler(X_test,  scaler)

    # SMOTE applied AFTER split to avoid data leakage
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # Task 5 — train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res)

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring="roc_auc")
    print(f"[train] CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Task 6 — evaluate
    metrics = compute_metrics(model, X_test_scaled, y_test)
    metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_roc_auc_std"]  = round(float(cv_scores.std()),  4)

    # Task 7 — save everything
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)          # models/trained_model.pkl
    save_scaler(scaler)                      # models/scaler.pkl
    save_feature_names(FEATURE_NAMES)        # models/feature_names.pkl
    save_json(metrics, METRICS_PATH)         # models/model_metrics.json

    print(f"[train] Metrics: {metrics}")
    
    # Generate evaluation plots
    print("[train] Generating evaluation plots...")
    plot_confusion_matrix(model, X_test_scaled, y_test, save=True)
    plot_feature_importance(model, FEATURE_NAMES, top_n=12, save=True)
    
    # Generate SHAP explainability plots
    print("[train] Generating SHAP explainability plots...")
    try:
        explainer = get_shap_explainer(model, X_train_scaled)
        plot_shap_summary(explainer, X_test_scaled, FEATURE_NAMES, save=True)
        print("[train] SHAP plots saved to outputs/reports/")
    except Exception as e:
        print(f"[train] Warning: Could not generate SHAP plots: {e}")
    
    return model, scaler, metrics

if __name__ == "__main__":
    train()