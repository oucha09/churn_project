import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from prediction import predict_churn


app = FastAPI(
    title="Churn Prediction API",
    description="API REST de prediction du churn client",
    version="2.0.0",
)


class ClientData(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[float] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None
    model_name: str = "xgboost"


def _model_dump(client):
    return client.model_dump() if hasattr(client, "model_dump") else client.dict()


def make_prediction(client_dict):
    model_name = client_dict.pop("model_name", "xgboost")
    if model_name not in {"xgboost", "catboost", "logistic_regression"}:
        raise ValueError(
            "model_name doit etre 'xgboost', 'catboost' ou 'logistic_regression'"
        )
    result = predict_churn(client_dict, model_name)
    result["model_used"] = model_name
    return result


@app.get("/")
def root():
    return {"message": "Churn Prediction API operationnelle", "version": "2.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": ["xgboost", "catboost", "logistic_regression"],
    }


@app.post("/predict")
def predict(client: ClientData):
    try:
        return make_prediction(_model_dump(client))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(clients: list[ClientData]):
    try:
        results = []
        for index, client in enumerate(clients):
            result = make_prediction(_model_dump(client))
            result["client_index"] = index
            results.append(result)
        return {
            "total": len(results),
            "churners": sum(result["prediction"] for result in results),
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
