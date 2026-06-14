from pathlib import Path

import joblib
import pandas as pd

from preprocessing import transform_clients
from retention import (
    customer_segment,
    enrich_with_retention_segments,
    retention_recommendation,
    risk_level,
)


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_model(model_name="xgboost"):
    model = joblib.load(MODELS_DIR / f"{model_name}_model.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    return model, preprocessor


def preprocess_single(client_dict, preprocessor=None):
    """Transforme un client brut avec exactement le pipeline d'entrainement."""
    return transform_clients(client_dict, preprocessor)


def predict_churn(client_dict, model_name="xgboost"):
    model, preprocessor = load_model(model_name)
    input_df = preprocess_single(client_dict, preprocessor)
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    level = risk_level(probability)
    segment = customer_segment(client_dict, probability)
    recommendation = retention_recommendation(client_dict, probability)

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": level,
        "customer_segment": segment,
        "recommendation": recommendation,
        "label": "CHURN" if prediction == 1 else "NO CHURN",
    }


def predict_batch(df_clients, model_name="xgboost"):
    model, preprocessor = load_model(model_name)
    X = transform_clients(df_clients, preprocessor)
    result = df_clients.copy()
    result["churn_prediction"] = model.predict(X)
    result["churn_probability"] = model.predict_proba(X)[:, 1].round(4)
    result = enrich_with_retention_segments(result, "churn_probability")
    return result.sort_values("churn_probability", ascending=False)
