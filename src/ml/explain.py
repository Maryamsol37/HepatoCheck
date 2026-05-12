from typing import Any
import matplotlib.pyplot as plt

try:
    import shap
    from src.utils.paths import REPORTS_DIR
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    import shap
    from src.utils.paths import REPORTS_DIR

def get_shap_explainer(model, X_train):
    return shap.TreeExplainer(model)

def explain_single(explainer, X_single, feature_names, top_n=5) -> dict[str, Any]:
    shap_values = explainer.shap_values(X_single)
    # index [1] = contributions toward "Possible Risk" class
    values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    contributions = dict(zip(feature_names, values.tolist()))
    top = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n])
    return {"shap_values": contributions, "top_contributors": top}

def plot_shap_summary(explainer, X_test, feature_names, save=True) -> None:
    shap_values = explainer.shap_values(X_test)
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(
        values,
        X_test,
        feature_names=feature_names,
        show=False,
        plot_size=(10, 7),
        max_display=len(feature_names),
    )
    plt.gcf().suptitle("SHAP Summary — Possible Risk Class", y=1.01)
    plt.gcf().tight_layout()
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()