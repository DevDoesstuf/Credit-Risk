"""
Models: a linear baseline and the tuned gradient-boosted tree.

Imbalance handling follows the "class weighting over synthetic oversampling"
recipe: SMOTE lifts the minority class to a moderate ratio, then class weights
finish balancing during training. SMOTE uses imbalanced-learn when it's
installed and falls back to a small self-contained implementation otherwise, so
the pipeline runs in a bare environment too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    HGB_PARAM_DIST, RANDOM_SEED, SEARCH_CV, SEARCH_N_ITER, SEARCH_SCORING,
    SMOTE_RATIO, USE_CLASS_WEIGHT,
)


# --------------------------------------------------------------------------- #
# Synthetic oversampling
# --------------------------------------------------------------------------- #
def _smote_fallback(X: np.ndarray, y: np.ndarray, ratio: float, seed: int):
    """Minimal SMOTE: interpolate between minority points and their k-NN.

    Used only when imbalanced-learn isn't available. NaNs are median-imputed
    for the neighbor search, then synthetic rows are appended to the originals.
    """
    rng = np.random.default_rng(seed)
    minority = X[y == 1]
    n_major = int((y == 0).sum())
    n_target = int(ratio * n_major)
    n_new = max(0, n_target - len(minority))
    if n_new == 0 or len(minority) < 6:
        return X, y

    col_median = np.nanmedian(minority, axis=0)
    filled = np.where(np.isnan(minority), col_median, minority)
    k = min(5, len(minority) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(filled)
    _, idx = nn.kneighbors(filled)

    base = rng.integers(0, len(minority), n_new)
    neigh = idx[base, rng.integers(1, k + 1, n_new)]
    gap = rng.random((n_new, 1))
    synth = filled[base] + gap * (filled[neigh] - filled[base])

    X_res = np.vstack([X, synth])
    y_res = np.concatenate([y, np.ones(n_new, dtype=int)])
    return X_res, y_res


def oversample(X: pd.DataFrame, y: pd.Series, ratio=SMOTE_RATIO, seed=RANDOM_SEED):
    """SMOTE the training set to the configured minority:majority ratio."""
    if ratio is None:
        return X, y
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore

        X_res, y_res = SMOTE(sampling_strategy=ratio, random_state=seed).fit_resample(X, y)
        return X_res, y_res
    except Exception:
        cols = X.columns
        X_res, y_res = _smote_fallback(X.to_numpy(dtype=float), y.to_numpy(), ratio, seed)
        return pd.DataFrame(X_res, columns=cols), pd.Series(y_res, name=y.name)


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def train_baseline(X_raw: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Logistic regression on raw columns only — the 'before' model."""
    pre = ColumnTransformer(
        [("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(X_raw.columns))]
    )
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_raw, y)
    return pipe


# --------------------------------------------------------------------------- #
# Tuned gradient-boosted tree
# --------------------------------------------------------------------------- #
def train_final(X: pd.DataFrame, y: pd.Series, n_iter=SEARCH_N_ITER, cv=SEARCH_CV):
    """SMOTE -> RandomizedSearchCV over a class-weighted HistGradientBoosting.

    HGB handles NaNs natively, so no imputation is needed. Returns the fitted
    RandomizedSearchCV (use `.best_estimator_` for the model).
    """
    X_res, y_res = oversample(X, y)

    base = HistGradientBoostingClassifier(
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
        class_weight="balanced" if USE_CLASS_WEIGHT else None,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=HGB_PARAM_DIST,
        n_iter=n_iter,
        scoring=SEARCH_SCORING,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_res, y_res)
    return search
