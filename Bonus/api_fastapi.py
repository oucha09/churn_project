import sys
import os

# ── Chemins absolus ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd

# ── Chargement des modèles ─────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_model(model_name):
    path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)

xgb_model    = load_model('xgboost')
cat_model    = load_model('catboost')
feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))

# ══════════════════════════════════════════════════════════════
# APPLICATION FASTAPI
# ══════════════════════════════════════════════════════════════
app = FastAPI(
    title="Churn Prediction API",
    description="API REST pour prédire le churn client avec XGBoost & CatBoost",
    version="1.0.0"
)

# ── Schéma d'entrée ────────────────────────────────────────────
class ClientData(BaseModel):
    gender: int                             # 0=Female, 1=Male
    SeniorCitizen: int                      # 0 ou 1
    Partner: int                            # 0=No, 1=Yes
    Dependents: int                         # 0=No, 1=Yes
    tenure: float                           # mois
    PhoneService: int                       # 0=No, 1=Yes
    PaperlessBilling: int                   # 0=No, 1=Yes
    MonthlyCharges: float                   # en $
    TotalCharges: float                     # en $
    InternetService_Fiber_optic: int = 0
    InternetService_No: int = 0
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    PaymentMethod_Credit_card_automatic: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaymentMethod_Mailed_check: int = 0
    model_name: Optional[str] = "xgboost"  # "xgboost" ou "catboost"

# ── Fonction de prédiction ─────────────────────────────────────
def make_prediction(client_dict: dict, model_name: str):
    model = xgb_model if model_name == "xgboost" else cat_model

    input_df = pd.DataFrame([client_dict])

    # Ajouter les colonnes manquantes avec 0
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_cols]

    prediction  = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    if probability >= 0.70:
        risk_level     = "Élevé"
        recommendation = "Contacter immédiatement. Proposer contrat annuel -20%, TechSupport offert 3 mois."
    elif probability >= 0.40:
        risk_level     = "Moyen"
        recommendation = "Envoyer enquête satisfaction. Proposer programme de fidélité."
    else:
        risk_level     = "Faible"
        recommendation = "Client stable. Maintenir la relation standard."

    return {
        "prediction"     : prediction,
        "label"          : "CHURN" if prediction == 1 else "NO CHURN",
        "probability"    : round(probability, 4),
        "risk_level"     : risk_level,
        "recommendation" : recommendation,
        "model_used"     : model_name
    }

# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message"   : "Churn Prediction API — opérationnelle ✅",
        "endpoints" : ["/predict", "/predict/batch", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status"  : "ok",
        "modeles" : ["xgboost", "catboost"],
        "features": len(feature_cols)
    }

@app.post("/predict")
def predict(client: ClientData):
    try:
        client_dict = client.dict()
        model_name  = client_dict.pop("model_name")
        result      = make_prediction(client_dict, model_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(clients: list[ClientData]):
    try:
        results = []
        for i, client in enumerate(clients):
            client_dict = client.dict()
            model_name  = client_dict.pop("model_name")
            result      = make_prediction(client_dict, model_name)
            result["client_index"] = i
            results.append(result)
        return {
            "total"    : len(results),
            "churners" : sum(r["prediction"] for r in results),
            "results"  : results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════
# Lancer avec :  uvicorn api_fastapi:app --reload
# Tester sur  :  http://127.0.0.1:8000/docs