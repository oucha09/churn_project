from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import load_and_preprocess
from training import (
    evaluate_model,
    plot_feature_importance,
    train_catboost,
    train_logistic_regression,
    train_xgboost,
)


DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"


def main():
    X_train, X_test, y_train, y_test, _, _, feature_names = load_and_preprocess(
        DATA_PATH
    )
    models = {
        "XGBoost": train_xgboost(X_train, y_train),
        "CatBoost": train_catboost(X_train, y_train),
        "Logistic Regression": train_logistic_regression(X_train, y_train),
    }

    metrics = {}
    for name, model in models.items():
        metrics[name] = evaluate_model(model, X_test, y_test, name)
        plot_feature_importance(model, feature_names, name)

    comparison = pd.DataFrame(metrics).T
    comparison.index.name = "model"
    comparison.to_csv(MODELS_DIR / "model_metrics.csv")
    print("\nComparaison finale")
    print(comparison.round(4).to_string())


if __name__ == "__main__":
    main()
