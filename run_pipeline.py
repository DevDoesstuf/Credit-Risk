"""
End-to-end training pipeline.

    python run_pipeline.py                 # full run (generates synthetic data if needed)
    python run_pipeline.py --regenerate    # force fresh synthetic data
    python run_pipeline.py --n-iter 8      # faster hyperparameter search

Steps: (1) ensure raw data, (2) SQLite + SQL feature engineering, (3) split,
(4) baseline logreg on raw cols, (5) SMOTE + tuned class-weighted HGB on
engineered cols, (6) pick a recall-oriented threshold, (7) metrics + ROC, (8)
save models and a metrics report the dashboard reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    BASELINE_PATH, METRICS_PATH, MODEL_PATH, RAW_CSV, RECALL_TARGET,
    SEARCH_N_ITER,
)
from src import database, evaluate, features, model, explain  # noqa: E402
from src.generate_data import main as generate_main  # noqa: E402


def ensure_data(regenerate: bool) -> None:
    if regenerate or not RAW_CSV.exists():
        print("== Generating synthetic loan data ==")
        generate_main()
    else:
        print(f"== Using existing raw data at {RAW_CSV} ==")


def main() -> None:
    ap = argparse.ArgumentParser(description="Credit risk training pipeline")
    ap.add_argument("--regenerate", action="store_true", help="force new synthetic data")
    ap.add_argument("--n-iter", type=int, default=SEARCH_N_ITER, help="RandomizedSearchCV iterations")
    args = ap.parse_args()

    ensure_data(args.regenerate)

    print("\n== SQLite load + SQL feature engineering ==")
    df = database.prepare()

    print("\n== Train / validation / test split ==")
    # Test is held out for reporting; validation is used only to place the
    # operating threshold, so the reported recall is on genuinely unseen data.
    X_train, X_test, y_train, y_test = features.split(df)
    X_tr, X_val, y_tr, y_val = features.holdout(X_train, y_train)
    print(f"train={len(X_tr):,}  val={len(X_val):,}  test={len(X_test):,}  "
          f"default_rate={y_tr.mean():.3%}")

    print("\n== Baseline: logistic regression on raw features ==")
    base = model.train_baseline(features.baseline_matrix(X_tr), y_tr)
    base_proba = base.predict_proba(features.baseline_matrix(X_test))[:, 1]

    print(f"\n== Final: SMOTE + tuned HGB (n_iter={args.n_iter}) ==")
    Xtr_m = features.model_matrix(X_tr)
    Xval_m = features.model_matrix(X_val)
    Xte_m = features.model_matrix(X_test)
    search = model.train_final(Xtr_m, y_tr, n_iter=args.n_iter)
    final = search.best_estimator_
    final_proba = final.predict_proba(Xte_m)[:, 1]
    print("best params:", json.dumps(search.best_params_))

    # Threshold picked on the validation set to hit the recall target.
    val_proba = final.predict_proba(Xval_m)[:, 1]
    thr = evaluate.threshold_for_recall(y_val, val_proba, RECALL_TARGET)

    base_metrics = evaluate.scores(y_test, base_proba, threshold=0.5)
    final_metrics = evaluate.scores(y_test, final_proba, threshold=thr)

    evaluate.save_roc(y_test, base_proba, final_proba)

    print("\n== Global feature importance (permutation) ==")
    n_imp = min(20_000, len(Xte_m))
    idx = np.random.RandomState(0).choice(len(Xte_m), n_imp, replace=False)
    importance = explain.global_importance(final, Xte_m.iloc[idx], y_test.iloc[idx], n_repeats=3)
    print(importance.round(4).to_string())

    # Persist everything the dashboard / reader needs.
    joblib.dump(base, BASELINE_PATH)
    joblib.dump(final, MODEL_PATH)
    report = {
        "baseline": base_metrics,
        "final": final_metrics,
        "recall_target": RECALL_TARGET,
        "best_params": search.best_params_,
        "feature_importance": importance.round(5).to_dict(),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "default_rate": round(float(df["SeriousDlqin2yrs"].mean()), 4),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 56)
    print("SUMMARY")
    print("=" * 56)
    print(f"Baseline AUC : {base_metrics['auc']:.3f}")
    print(f"Final AUC    : {final_metrics['auc']:.3f}")
    print(f"Final recall : {final_metrics['recall']:.3f}  "
          f"(target {RECALL_TARGET:.2f}, threshold {thr:.3f})")
    print(f"Final prec.  : {final_metrics['precision']:.3f}")
    print(f"Saved model  -> {MODEL_PATH}")
    print(f"Saved report -> {METRICS_PATH}")
    print(f"Saved ROC    -> reports/roc_curve.png")


if __name__ == "__main__":
    main()
