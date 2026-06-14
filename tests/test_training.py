import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from training import (
    evaluate_model,
    plot_feature_importance,
    train_logistic_regression,
)


def test_evaluation_and_feature_importance_return_complete_results(tmp_path):
    X = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    y = pd.Series([0, 0, 1, 1])
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

    metrics = evaluate_model(model, X, y, "TestModel", output_dir=tmp_path)
    ranking = plot_feature_importance(
        model, X.columns, "TestModel", top_n=2, output_dir=tmp_path
    )

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert set(ranking.columns) == {"feature", "importance"}


def test_logistic_regression_can_predict_probabilities(tmp_path, monkeypatch):
    X = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    y = pd.Series([0, 0, 1, 1])
    monkeypatch.setattr("training.MODELS_DIR", tmp_path)

    model = train_logistic_regression(X, y)

    assert model.predict_proba(X).shape == (4, 2)
