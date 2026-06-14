from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def evaluate_model(
    model, X_test, y_test, model_name="Model", show=False, output_dir=MODELS_DIR
):
    """Calcule et sauvegarde les metriques, la matrice de confusion et la ROC."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'=' * 40}\n  {model_name}\n{'=' * 40}")
    for metric, value in metrics.items():
        print(f"{metric:10}: {value:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix - {model_name}")
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{model_name.replace(' ', '_')}_eval.png")
    if show:
        plt.show()
    plt.close(fig)
    return metrics


def train_xgboost(X_train, y_train):
    """Entraine XGBoost avec des hyperparametres de base explicites."""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=3,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "xgboost_model.pkl")
    print("XGBoost sauvegarde")
    return model


def train_catboost(X_train, y_train):
    """Entraine CatBoost avec des hyperparametres de base explicites."""
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        class_weights=[1, 3],
        verbose=50,
        random_seed=42,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "catboost_model.pkl")
    print("CatBoost sauvegarde")
    return model


def train_logistic_regression(X_train, y_train):
    """Entraine une regression logistique servant de modele de reference."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / "logistic_regression_model.pkl")
    print("Logistic Regression sauvegardee")
    return model


def get_feature_importances(model):
    """Retourne une importance comparable pour arbres et regression logistique."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        return abs(model.named_steps["classifier"].coef_[0])
    raise ValueError("Ce modele ne fournit pas d'importance de variables")


def plot_feature_importance(
    model,
    feature_names,
    model_name="XGBoost",
    top_n=15,
    show=False,
    output_dir=MODELS_DIR,
):
    """Classe et sauvegarde les variables les plus influentes."""
    file_name = model_name.replace(" ", "_")
    ranking = pd.DataFrame(
        {"feature": feature_names, "importance": get_feature_importances(model)}
    ).sort_values("importance", ascending=False)
    output_dir = Path(output_dir)
    ranking.to_csv(output_dir / f"{file_name}_feature_importance.csv", index=False)

    top = ranking.head(min(top_n, len(ranking))).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["importance"], color="steelblue")
    ax.set_title(f"Top {len(top)} Features Importantes - {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / f"{file_name}_feature_importance.png")
    if show:
        plt.show()
    plt.close(fig)
    return ranking
