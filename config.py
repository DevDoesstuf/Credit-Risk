"""
Central configuration for the Credit Risk Prediction Pipeline.

Everything that another module might want to tweak (paths, column groups,
imbalance handling, the hyperparameter search space) lives here so the rest
of the code stays declarative.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SQL_DIR = ROOT / "sql"

RAW_CSV = DATA_DIR / "loans_raw.csv"
DB_PATH = DATA_DIR / "credit_risk.db"
FEATURE_SQL = SQL_DIR / "feature_engineering.sql"

MODEL_PATH = MODEL_DIR / "credit_risk_hgb.joblib"
BASELINE_PATH = MODEL_DIR / "baseline_logreg.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
ROC_PATH = REPORT_DIR / "roc_curve.png"

for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
# Column names mirror the Kaggle "Give Me Some Credit" dataset (150k rows),
# which is what the pipeline was originally built against. The synthetic
# generator produces the same schema so the project runs end-to-end with no
# external download; drop the real cs-training.csv in as data/loans_raw.csv
# and everything downstream works unchanged.
TARGET = "SeriousDlqin2yrs"

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Columns produced by sql/feature_engineering.sql (window-function features).
ENGINEERED_FEATURES = [
    "delinquency_score",       # weighted sum of the three past-due counts
    "peer_delinquency_gap",    # borrower score minus the mean for their age band
    "income_percentile",       # NTILE(100) over MonthlyIncome
    "risk_tier",               # NTILE(5) over a composite risk score (1=safest,5=riskiest)
]

# Model input = raw features + engineered features (minus columns we bucket away).
MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

# Direction of effect used for plain-language attribution in explain.py.
# +1  -> higher value increases estimated risk
# -1  -> higher value decreases estimated risk
FEATURE_DIRECTION = {
    "RevolvingUtilizationOfUnsecuredLines": +1,
    "age": -1,
    "NumberOfTime30-59DaysPastDueNotWorse": +1,
    "DebtRatio": +1,
    "MonthlyIncome": -1,
    "NumberOfOpenCreditLinesAndLoans": 0,
    "NumberOfTimes90DaysLate": +1,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": +1,
    "NumberOfDependents": +1,
    "delinquency_score": +1,
    "peer_delinquency_gap": +1,
    "income_percentile": -1,
    "risk_tier": +1,
}

# Human-readable labels for the dashboard / explanations.
FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "revolving credit utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "times 30-59 days past due",
    "DebtRatio": "debt ratio",
    "MonthlyIncome": "monthly income",
    "NumberOfOpenCreditLinesAndLoans": "open credit lines",
    "NumberOfTimes90DaysLate": "times 90+ days late",
    "NumberRealEstateLoansOrLines": "real-estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "times 60-89 days past due",
    "NumberOfDependents": "dependents",
    "delinquency_score": "overall delinquency score",
    "peer_delinquency_gap": "delinquency vs. same-age peers",
    "income_percentile": "income percentile",
    "risk_tier": "risk tier",
}

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
N_ROWS = 150_000
TARGET_DEFAULT_RATE = 0.05

# --------------------------------------------------------------------------- #
# Imbalance handling
# --------------------------------------------------------------------------- #
# The resume line describes "class weighting over synthetic oversampling":
# SMOTE lifts the minority class part-way, then class weights finish the job.
# SMOTE_RATIO is the minority:majority ratio SMOTE targets (0.30 == minority
# becomes 30% the size of the majority). Set to None to skip oversampling and
# rely on class weights alone.
SMOTE_RATIO = 0.30
USE_CLASS_WEIGHT = True

# --------------------------------------------------------------------------- #
# Train / test split
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.20

# Threshold chosen on the validation set to hit the recall target the business
# cares about (catching a large share of true defaults).
RECALL_TARGET = 0.68

# --------------------------------------------------------------------------- #
# HistGradientBoosting hyperparameter search space (RandomizedSearchCV)
# --------------------------------------------------------------------------- #
HGB_PARAM_DIST = {
    "learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
    "max_iter": [200, 350, 500, 700],
    "max_leaf_nodes": [15, 31, 63, 127],
    "max_depth": [None, 4, 6, 8, 12],
    "min_samples_leaf": [20, 50, 100, 200],
    "l2_regularization": [0.0, 0.1, 1.0, 5.0],
    "max_bins": [128, 255],
}

SEARCH_N_ITER = 25
SEARCH_CV = 3
SEARCH_SCORING = "roc_auc"
