"""
SQLite loading + SQL feature engineering.

Flow:
  1. load raw CSV into the `loans_raw` table,
  2. execute sql/feature_engineering.sql (window functions) -> `loans_features`,
  3. hand the finished table back to Python as a DataFrame.

Keeping the ranking/aggregation in SQL means it's reproducible (the same
script defines the features every run) and pushdown-efficient (the engine does
the sort/partition work once over the whole table).
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from config import DB_PATH, FEATURE_SQL, RAW_CSV


def load_raw_to_sqlite(csv_path=RAW_CSV, db_path=DB_PATH) -> None:
    """Load the raw loan CSV into the loans_raw table (replacing any prior copy)."""
    df = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("loans_raw", conn, if_exists="replace", index=False)
    print(f"Loaded {len(df):,} rows into loans_raw ({db_path.name})")


def build_features(db_path=DB_PATH, sql_path=FEATURE_SQL) -> None:
    """Run the feature-engineering SQL to (re)build loans_features."""
    sql = sql_path.read_text()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(sql)
    print("Built loans_features via window-function SQL")


def read_features(db_path=DB_PATH) -> pd.DataFrame:
    """Read the engineered feature table back into pandas."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM loans_features", conn)


def prepare(csv_path=RAW_CSV, db_path=DB_PATH) -> pd.DataFrame:
    """End-to-end: raw CSV -> SQLite -> engineered features -> DataFrame."""
    load_raw_to_sqlite(csv_path, db_path)
    build_features(db_path)
    return read_features(db_path)


if __name__ == "__main__":
    frame = prepare()
    print(frame.head())
    print(frame[["risk_tier", "income_percentile", "delinquency_score"]].describe())
