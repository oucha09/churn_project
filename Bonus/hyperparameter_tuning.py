import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import load_and_preprocess


DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"


def build_searches(quick=False, n_jobs=1):
    """Construit RandomizedSearchCV pour les arbres et GridSearchCV pour la baseline."""
    iterations = 4 if quick else 20
    cv = 3 if quick else 5

    xgb_search = RandomizedSearchCV(
        xgb.XGBClassifier(scale_pos_weight=3, eval_metric="logloss", random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
        },
        n_iter=iterations,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        random_state=42,
        verbose=1,
    )
    cat_search = RandomizedSearchCV(
        CatBoostClassifier(
            loss_function="Logloss", class_weights=(1, 3), verbose=0, random_seed=42
        ),
        {
            "iterations": [200, 300, 500],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5, 7],
        },
        n_iter=iterations,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        random_state=42,
        verbose=1,
    )
    logistic_search = GridSearchCV(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=42
                    ),
                ),
            ]
        ),
        {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear", "lbfgs"],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        verbose=1,
    )
    return {
        "xgboost": xgb_search,
        "catboost": cat_search,
        "logistic_regression": logistic_search,
    }


def run_tuning(quick=False, n_jobs=1):
    X_train, X_test, y_train, y_test, *_ = load_and_preprocess(DATA_PATH)
    summaries = []

    for name, search in build_searches(quick, n_jobs=n_jobs).items():
        print(f"\nOptimisation {name}...")
        search.fit(X_train, y_train)
        estimator = search.best_estimator_
        test_roc_auc = roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1])
        joblib.dump(estimator, MODELS_DIR / f"{name}_optimized_model.pkl")

        summary = {
            "model": name,
            "cv_roc_auc": search.best_score_,
            "test_roc_auc": test_roc_auc,
            "best_params": search.best_params_,
        }
        summaries.append(summary)
        pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
            MODELS_DIR / f"{name}_tuning_results.csv", index=False
        )

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(MODELS_DIR / "tuning_summary.csv", index=False)
    with open(MODELS_DIR / "best_hyperparameters.json", "w", encoding="utf-8") as file:
        json.dump(
            {row["model"]: row["best_params"] for row in summaries},
            file,
            indent=2,
        )
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Recherche courte pour validation locale rapide.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Nombre de processus paralleles utilises par sklearn.",
    )
    args = parser.parse_args()
    comparison = run_tuning(quick=args.quick, n_jobs=args.n_jobs)
    print("\nResultats optimisation")
    print(comparison[["model", "cv_roc_auc", "test_roc_auc"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
