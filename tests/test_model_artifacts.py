import os
import json
from pathlib import Path


def test_best_pipeline_exists():
    assert Path('models/best_pipeline.pkl').exists(), 'best_pipeline.pkl is missing in models/'


def test_tuning_metrics_threshold():
    fp = Path('models/tuning_metrics.json')
    assert fp.exists(), 'tuning_metrics.json missing'
    data = json.loads(fp.read_text(encoding='utf-8'))
    val = data.get('nested_cv_mean_roc_auc') or data.get('nested_cv_mean_roc_auc'.lower())
    # fallback: try keys
    if val is None:
        # try to find numeric value in file
        for k in data:
            if isinstance(data[k], (int, float)):
                val = data[k]
                break
    assert val is not None, 'Could not find nested CV ROC-AUC in tuning_metrics.json'
    assert float(val) >= 0.85, f'nested CV ROC-AUC {val} below threshold 0.85'