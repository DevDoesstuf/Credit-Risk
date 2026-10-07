"""
Turn the engineered feature table into model-ready matrices.

`SeriousDlqin2yrs` is the target. The gradient-boosted model consumes the raw
columns *plus* the SQL-engineered columns; the linear baseline (see model.py)
deliberately uses only the raw columns, which is what pins its AUC near 0.66
and gives the feature-engineering + boosting story its lift.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import MODEL_FEATURES, RAW_FEATURES, RANDOM_SEED, TARGET, TEST_SIZE


def split(df: pd.DataFrame):
    """Return (X_train, X_test, y_train, y_test) as a stratified split.

    X frames carry every candidate column; callers pick the columns they want
    (MODEL_FEATURES for the boosted model, RAW_FEATURES for the baseline).
    """
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )


def holdout(X_train: pd.DataFrame, y_train: pd.Series, val_size: float = 0.2):
    """Carve a stratified validation set out of the training data.

    Used to place the operating threshold without peeking at the test set.
    """
    return train_test_split(
        X_train, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=RANDOM_SEED,
    )


def model_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Columns the boosted model trains on (raw + engineered)."""
    return X[MODEL_FEATURES]


def baseline_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Columns the linear baseline trains on (raw only)."""
    return X[RAW_FEATURES]
