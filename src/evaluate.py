"""
Evaluation utilities: AUC / recall / precision, a threshold chosen to hit the
business recall target, and a saved ROC curve comparing baseline vs. final.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve,
)

from config import RECALL_TARGET, ROC_PATH


def threshold_for_recall(y_true, proba, target=RECALL_TARGET) -> float:
    """Lowest probability cutoff that still catches `target` share of defaults."""
    order = np.argsort(-proba)
    sorted_true = np.asarray(y_true)[order]
    total_pos = sorted_true.sum()
    if total_pos == 0:
        return 0.5
    cum_tp = np.cumsum(sorted_true)
    recall = cum_tp / total_pos
    hit = np.argmax(recall >= target)
    return float(proba[order][hit])


def scores(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "threshold": round(float(threshold), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_roc(y_true, proba_baseline, proba_final, path=ROC_PATH) -> None:
    plt.figure(figsize=(6, 6))
    for proba, label in [(proba_baseline, "Baseline (logreg, raw)"),
                         (proba_final, "Tuned HGB (engineered)")]:
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.plot(fpr, tpr, label=f"{label} — AUC {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Credit risk — ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved ROC curve to {path}")
