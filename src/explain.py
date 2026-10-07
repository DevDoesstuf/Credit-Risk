"""
Plain-language feature attribution for a single borrower.

Global signal comes from permutation importance (which features actually move
this model's AUC). For one borrower, each important feature is turned into a
sentence by combining:
  * how extreme the borrower's value is (its percentile in the population), and
  * the known direction of effect (config.FEATURE_DIRECTION),
so a non-technical reader gets "why" rather than a coefficient dump.

This is intentionally model-agnostic and dependency-free. If you later add
`shap`, TreeExplainer values can drop straight into `attribute_borrower` in
place of the percentile heuristic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import FEATURE_DIRECTION, FEATURE_LABELS, MODEL_FEATURES


def global_importance(model, X, y, n_repeats=5, seed=42) -> pd.Series:
    """Permutation importance over the model's features, sorted descending."""
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def _percentile(value, column_values) -> float:
    col = np.asarray(column_values, dtype=float)
    col = col[~np.isnan(col)]
    if len(col) == 0 or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    return float((col < value).mean() * 100)


def _phrase(pct: float) -> str:
    """Plain-language magnitude word for where a value sits in the population."""
    if pct >= 90:
        return "very high"
    if pct >= 70:
        return "high"
    if pct <= 10:
        return "very low"
    if pct <= 30:
        return "low"
    return "moderate"


def attribute_borrower(borrower: pd.Series, population: pd.DataFrame,
                       importance: pd.Series, top_k: int = 4) -> list[dict]:
    """Return the top_k drivers of this borrower's risk as plain-language items.

    Each item: {feature, label, value, percentile, effect ('raises'/'lowers'),
    sentence}. Only features with a known non-zero direction are surfaced.
    """
    reasons = []
    for feature in importance.index:
        if feature not in MODEL_FEATURES:
            continue
        direction = FEATURE_DIRECTION.get(feature, 0)
        if direction == 0:
            continue
        value = borrower.get(feature, np.nan)
        pct = _percentile(value, population[feature])
        if np.isnan(pct):
            continue

        # Does this borrower's value push risk up or down?
        above_median = pct >= 50
        raises = (above_median and direction > 0) or (not above_median and direction < 0)
        level = _phrase(pct)
        label = FEATURE_LABELS.get(feature, feature)
        verb = "raises" if raises else "lowers"
        sentence = (f"{level.capitalize()} {label} "
                    f"({pct:.0f}th pct) {verb} estimated risk")

        reasons.append({
            "feature": feature,
            "label": label,
            "value": None if (isinstance(value, float) and np.isnan(value)) else value,
            "percentile": round(pct, 1),
            "effect": "raises" if raises else "lowers",
            "sentence": sentence,
        })
        if len(reasons) >= top_k:
            break
    return reasons
