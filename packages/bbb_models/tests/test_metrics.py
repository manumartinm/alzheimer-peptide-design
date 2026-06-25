import numpy as np

from bbb_classifier.train.metrics import classification_metrics


def test_metrics_keys_present():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    m = classification_metrics(y, p)
    for key in ["roc_auc", "pr_auc", "mcc", "precision", "sensitivity", "brier"]:
        assert key in m
