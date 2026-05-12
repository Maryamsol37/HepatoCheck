import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score
    )
    from src.utils.paths import REPORTS_DIR
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score
    )
    from src.utils.paths import REPORTS_DIR

def compute_metrics(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba)), 4),
    }
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "Possible Risk"]))
    return metrics

def plot_confusion_matrix(model, X_test, y_test, save=True) -> None:
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low Risk", "Possible Risk"],
                yticklabels=["Low Risk", "Possible Risk"])
    plt.title("Confusion Matrix"); plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, top_n=12, save=True) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
    plt.title("Feature Importances"); plt.xlabel("Importance Score")
    plt.tight_layout()
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
        plt.close()
    else:
        plt.show()