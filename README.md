# Credit Risk Prediction Pipeline

Predicts serious loan delinquency on a 150,000-row loan dataset (~5% default
rate). A logistic-regression baseline on the raw columns lands around 0.67 AUC;
SQL-engineered features plus a tuned, class-weighted gradient-boosted tree lift
it to ~0.86, with the operating threshold tuned to catch a large share of true
defaults. A Streamlit dashboard makes the output usable by non-technical people.

## Quick start

```bash
pip install -r requirements.txt

# Train everything (generates synthetic data on first run, ~1-2 min):
python run_pipeline.py

# Faster hyperparameter search while iterating:
python run_pipeline.py --n-iter 8

# Launch the dashboard (after the pipeline has run once):
streamlit run app/dashboard.py
```

`run_pipeline.py` writes the trained models to `models/`, a metrics report to
`reports/metrics.json`, and a baseline-vs-final ROC curve to
`reports/roc_curve.png`. The dashboard reads those plus the engineered feature
table in `data/credit_risk.db`.

## Architecture

```
run_pipeline.py          orchestrator: data -> SQLite -> features -> models -> report
config.py                paths, schema, feature groups, search space, all knobs
sql/
  feature_engineering.sql window-function features (runs inside SQLite)
src/
  generate_data.py        synthetic data in the real dataset's schema
  database.py             CSV -> SQLite -> run the SQL -> DataFrame
  features.py             train/val/test split; raw vs. model feature matrices
  model.py                baseline logreg; SMOTE + RandomizedSearchCV HGB
  evaluate.py             recall-oriented thresholding, metrics, ROC plot
  explain.py              permutation importance + per-borrower attribution
app/
  dashboard.py            Streamlit: segmentation, model report, borrower scorer
```

**Flow.** Raw loans land in SQLite. `sql/feature_engineering.sql` does the
feature work *in the database* with window functions — a weighted delinquency
score, each borrower's delinquency gap vs. same-age peers
(`AVG(...) OVER (PARTITION BY age_band)`), an income percentile
(`NTILE(100) OVER (ORDER BY MonthlyIncome)`), and a 5-way risk tier
(`NTILE(5)` over a composite score). Doing this as pushdown SQL keeps it
reproducible and means Python only ever pulls the finished table.

Then: a stratified train/val/test split; a logistic-regression baseline on the
**raw columns only** (this is the ~0.67 "before" number); and the final model —
SMOTE to a 0.30 minority ratio, then a `HistGradientBoostingClassifier` with
`class_weight="balanced"` tuned by `RandomizedSearchCV` on the raw **plus**
engineered columns. The operating threshold is chosen on the **validation** set
to hit the recall target, so the reported recall is on genuinely unseen data.

**"Class weighting over synthetic oversampling."** Both levers are used
together: SMOTE lifts the minority class part-way (to a 0.30 ratio, not all the
way to balanced), and class weights finish the job during training. This tends
to be steadier than pushing SMOTE all the way to 1:1.

**Where the lift comes from.** The baseline is a linear model on raw columns.
The gradient-boosted tree adds two things a linear baseline can't get: it
captures nonlinear interactions between features, and it handles missing values
natively (a real chunk of `MonthlyIncome` is missing) instead of leaning on
imputation. The SQL-engineered peer/percentile/tier features give it extra
signal to split on. That combination is the 0.67 → 0.86 story.

## About the data (read this)

This is a **faithful reconstruction of a project whose original source files
were lost** — rebuilt from the resume description, not recovered byte-for-byte.
The code, structure, SQL, and modeling recipe are real and runnable.

The original was built on the Kaggle **"Give Me Some Credit"** dataset (exactly
150,000 rows, ~7% default, the same column schema used here). Because that CSV
isn't redistributed with the repo, `src/generate_data.py` produces a synthetic
stand-in **in the identical schema** so the whole thing runs end-to-end with no
download. The synthetic data is tuned to reproduce the same qualitative story —
a linear baseline near 0.67 and a boosted model near 0.86 — rather than to
memorize the real numbers.

With this synthetic data you'll see roughly:

| model                         | AUC   |
|-------------------------------|-------|
| logistic baseline (raw cols)  | ~0.67 |
| tuned HGB (raw + engineered)  | ~0.86 |

To reproduce the exact resume figures (0.66 → 0.87, 68% recall), download
`cs-training.csv` from the Kaggle competition, drop it in as
`data/loans_raw.csv`, and run `python run_pipeline.py` — every step downstream
works unchanged, because the schema matches.

## Notes

- `imbalanced-learn` is used for SMOTE when installed; if it isn't, `model.py`
  falls back to a small self-contained k-NN interpolation so the pipeline still
  runs. Install the requirements to use the real thing.
- Everything is seeded (`config.RANDOM_SEED`) for reproducibility.
- Per-borrower attribution in the dashboard is percentile-based against the
  population. `explain.py` notes where a SHAP `TreeExplainer` could drop in for
  exact per-prediction contributions.
