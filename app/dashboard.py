"""
Streamlit dashboard for the Credit Risk Prediction Pipeline.

    streamlit run app/dashboard.py

Three pages:
  * Overview      - default rates segmented by risk tier, age band, and income
  * Model         - AUC, ROC curve, confusion matrix, baseline vs. final
  * Score a borrower - pick or edit a borrower and get a probability plus
                       plain-language reasons behind the score

The dashboard reads the SQL-engineered feature table straight from SQLite and
loads the trained model / metrics report that run_pipeline.py wrote. Run the
pipeline first so those artifacts exist:

    python run_pipeline.py

The heavy lifting (loading data, segmenting, scoring) lives in plain functions
that don't import streamlit, so the logic can be tested on its own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make the project root importable whether run from the repo root or app/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import database, explain, features  # noqa: E402


# --------------------------------------------------------------------------- #
# Data / model loading (no streamlit here so it's testable)
# --------------------------------------------------------------------------- #
def load_features() -> pd.DataFrame:
    """The SQL-engineered feature table, straight from SQLite."""
    return database.read_features()


def load_model():
    return joblib.load(config.MODEL_PATH)


def load_baseline():
    return joblib.load(config.BASELINE_PATH)


def load_metrics() -> dict:
    return json.loads(config.METRICS_PATH.read_text())


def _income_band(df: pd.DataFrame) -> pd.Series:
    """Bucket monthly income into five labelled bands by quintile.

    NaN incomes (a real slice of the data) get their own band so they show up
    in the segmentation instead of silently vanishing.
    """
    income = df["MonthlyIncome"]
    known = income.dropna()
    # Quintile edges from the observed distribution.
    edges = known.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy().copy()
    edges[0], edges[-1] = -np.inf, np.inf
    labels = ["lowest 20%", "20-40%", "40-60%", "60-80%", "top 20%"]
    band = pd.cut(income, bins=edges, labels=labels, include_lowest=True)
    band = band.cat.add_categories(["unknown"]).fillna("unknown")
    return band


def default_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Default rate and row count grouped by a column, ready to chart."""
    g = df.groupby(column, observed=True)[config.TARGET]
    out = pd.DataFrame({"default_rate": g.mean(), "borrowers": g.size()})
    return out.reset_index()


def score_borrower(model, borrower: pd.Series, population: pd.DataFrame,
                   importance: pd.Series, top_k: int = 4):
    """Probability of default plus the top plain-language drivers.

    `borrower` is a row that already carries the engineered columns (i.e. a row
    from the features table), so we just line up MODEL_FEATURES in order.
    """
    x = borrower.reindex(config.MODEL_FEATURES).to_frame().T
    x = x.apply(pd.to_numeric, errors="coerce")
    proba = float(model.predict_proba(x)[:, 1][0])
    reasons = explain.attribute_borrower(
        borrower, population[config.MODEL_FEATURES], importance, top_k=top_k
    )
    return proba, reasons


def importance_series(metrics: dict) -> pd.Series:
    """Permutation importance from the metrics report, ordered for attribution."""
    imp = pd.Series(metrics.get("feature_importance", {}), dtype=float)
    if imp.empty:
        # Fall back to equal weights so attribution still has an ordering.
        imp = pd.Series({f: 1.0 for f in config.MODEL_FEATURES})
    return imp.sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def _require_streamlit():
    try:
        import streamlit as st  # noqa: F401
        return st
    except ModuleNotFoundError:  # pragma: no cover
        raise SystemExit(
            "streamlit isn't installed. Install requirements first:\n"
            "    pip install -r requirements.txt\n"
            "then run:\n"
            "    streamlit run app/dashboard.py"
        )


def _check_artifacts(st) -> bool:
    missing = [p for p in (config.DB_PATH, config.MODEL_PATH, config.METRICS_PATH)
               if not Path(p).exists()]
    if missing:
        st.error("Missing pipeline artifacts. Run the pipeline first:")
        st.code("python run_pipeline.py", language="bash")
        st.write("Missing:", ", ".join(str(m) for m in missing))
        return False
    return True


def page_overview(st, df: pd.DataFrame):
    st.header("Portfolio default rates")
    st.caption(
        f"{len(df):,} borrowers · overall default rate "
        f"{df[config.TARGET].mean():.2%}"
    )

    df = df.copy()
    df["income_band"] = _income_band(df)

    segments = [
        ("risk_tier", "By risk tier (1 = safest, 5 = riskiest)"),
        ("age_band", "By age band"),
        ("income_band", "By income band"),
    ]
    for column, title in segments:
        st.subheader(title)
        seg = default_rate_by(df, column).set_index(column)
        st.bar_chart(seg["default_rate"])
        seg_display = seg.copy()
        seg_display["default_rate"] = (seg_display["default_rate"] * 100).round(2)
        seg_display = seg_display.rename(columns={"default_rate": "default_rate_%"})
        st.dataframe(seg_display, use_container_width=True)


def page_model(st, metrics: dict):
    st.header("Model performance")

    base, final = metrics["baseline"], metrics["final"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline AUC", f"{base['auc']:.3f}")
    c2.metric("Final AUC", f"{final['auc']:.3f}",
              delta=f"+{final['auc'] - base['auc']:.3f}")
    c3.metric("Final recall", f"{final['recall']:.1%}",
              help=f"target {metrics['recall_target']:.0%}, "
                   f"threshold {final['threshold']:.3f}")

    if Path(config.ROC_PATH).exists():
        st.subheader("ROC curve — baseline vs. final")
        st.image(str(config.ROC_PATH), use_container_width=True)

    st.subheader("Confusion matrix (final model, test set)")
    cm = final["confusion"]
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["actual: repaid", "actual: default"],
        columns=["predicted: repaid", "predicted: default"],
    )
    st.dataframe(cm_df, use_container_width=True)
    st.caption(
        f"Precision {final['precision']:.1%} · recall {final['recall']:.1%}. "
        "The threshold is tuned for recall — catching defaults matters more "
        "than avoiding false alarms here, so precision is intentionally traded "
        "down."
    )

    st.subheader("What the model leans on (permutation importance)")
    imp = importance_series(metrics)
    imp = imp[imp > 0]
    st.bar_chart(imp)


def page_score(st, df: pd.DataFrame, model, metrics: dict):
    st.header("Score a borrower")
    st.caption(
        "Pick a borrower from the portfolio, then adjust any field to see how "
        "the score moves. Reasons are ranked by how much the model relies on "
        "each feature."
    )

    importance = importance_series(metrics)
    population = df

    max_idx = len(df) - 1
    default_loan = int(df.iloc[max_idx // 2].get("loan_id", 0))
    loan_id = st.number_input(
        "Loan ID", min_value=int(df["loan_id"].min()),
        max_value=int(df["loan_id"].max()), value=default_loan, step=1,
    )
    row = df[df["loan_id"] == loan_id]
    if row.empty:
        st.warning("No borrower with that loan ID.")
        return
    borrower = row.iloc[0].copy()

    st.subheader("Borrower inputs")
    editable = [
        "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
        "MonthlyIncome", "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate", "NumberOfOpenCreditLinesAndLoans",
        "NumberOfDependents",
    ]
    cols = st.columns(2)
    for i, feat in enumerate(editable):
        raw_val = borrower.get(feat, np.nan)
        val = 0.0 if pd.isna(raw_val) else float(raw_val)
        borrower[feat] = cols[i % 2].number_input(
            config.FEATURE_LABELS.get(feat, feat), value=val
        )

    proba, reasons = score_borrower(model, borrower, population, importance)

    st.subheader("Result")
    st.metric("Estimated probability of default", f"{proba:.1%}")
    tier = int(borrower.get("risk_tier", 0))
    st.write(f"SQL risk tier: **{tier}** (1 = safest, 5 = riskiest)")

    st.subheader("Why")
    if not reasons:
        st.write("No attributable drivers for this borrower.")
    for r in reasons:
        arrow = "🔺" if r["effect"] == "raises" else "🔻"
        st.write(f"{arrow} {r['sentence']}")


def main():  # pragma: no cover - exercised only under `streamlit run`
    st = _require_streamlit()
    st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
    st.title("Credit Risk Prediction Pipeline")

    if not _check_artifacts(st):
        return

    @st.cache_data
    def _features():
        return load_features()

    @st.cache_resource
    def _model():
        return load_model()

    @st.cache_data
    def _metrics():
        return load_metrics()

    df = _features()
    model = _model()
    metrics = _metrics()

    page = st.sidebar.radio(
        "Page", ["Overview", "Model performance", "Score a borrower"]
    )
    if page == "Overview":
        page_overview(st, df)
    elif page == "Model performance":
        page_model(st, metrics)
    else:
        page_score(st, df, model, metrics)


if __name__ == "__main__":
    main()
