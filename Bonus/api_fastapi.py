import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
MODELS_DIR = Path(BASE_DIR) / "models"
DATA_PATH = Path(BASE_DIR) / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
METRICS_PATH = MODELS_DIR / "model_metrics.csv"
SUPPORTED_MODELS = ("xgboost", "catboost", "logistic_regression")

from prediction import predict_batch as predict_churn_batch
from prediction import predict_churn


app = FastAPI(
    title="Churn Prediction API",
    description="API REST de prediction du churn client",
    version="2.0.0",
)

production_origin = os.getenv("FRONTEND_ORIGIN")
allowed_origins = [
    "http://localhost",
    "http://127.0.0.1",
]
if production_origin:
    allowed_origins.append(production_origin.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def _normalize_model_name(model_name):
    normalized = model_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in SUPPORTED_MODELS:
        raise ValueError(
            "model doit etre 'xgboost', 'catboost' ou 'logistic_regression'"
        )
    return normalized


def make_prediction(client_dict, requested_model=None):
    body_model = client_dict.pop("model_name", "xgboost")
    model_name = _normalize_model_name(requested_model or body_model)
    result = predict_churn(client_dict, model_name)
    return {
        "model_used": model_name,
        "prediction": "Churn" if result["prediction"] == 1 else "No Churn",
        "probability": result["probability"],
        "segment": result["customer_segment"],
        "retention_recommendation": result["recommendation"],
        # Existing response fields are preserved for current API consumers.
        "prediction_value": result["prediction"],
        "risk_level": result["risk_level"],
        "customer_segment": result["customer_segment"],
        "recommendation": result["recommendation"],
        "label": result["label"],
    }


def _load_csv(path):
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Fichier introuvable: {path.name}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Impossible de lire {path.name}: {exc}"
        ) from exc


def _percentage_records(series, label_column):
    result = series.rename("churn_rate").reset_index()
    result["churn_rate"] = (result["churn_rate"] * 100).round(2)
    return result.rename(columns={result.columns[0]: label_column}).to_dict("records")


@lru_cache(maxsize=len(SUPPORTED_MODELS))
def _all_customer_predictions(model_name):
    data = _load_csv(DATA_PATH)
    return predict_churn_batch(data, model_name)


@app.get("/")
def root():
    return {"message": "Churn Prediction API operationnelle", "version": "2.0.0"}


@app.get("/health")
def health():
    models_loaded = [
        model_name
        for model_name in SUPPORTED_MODELS
        if (MODELS_DIR / f"{model_name}_model.pkl").exists()
    ]
    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "models": models_loaded,
    }


@app.post("/predict")
def predict(
    client: ClientData,
    model: Optional[str] = Query(default=None, description="Modele ML a utiliser"),
):
    try:
        return make_prediction(_model_dump(client), model)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(
    clients: list[ClientData],
    model: Optional[str] = Query(default=None, description="Modele ML a utiliser"),
):
    try:
        model_name = _normalize_model_name(model or "xgboost")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    results = []
    errors = []
    for index, client in enumerate(clients):
        try:
            result = make_prediction(_model_dump(client), model_name)
            result["client_index"] = index
            results.append(result)
        except Exception as exc:
            errors.append({"client_index": index, "detail": str(exc)})
    return {
        "model_used": model_name,
        "total": len(clients),
        "successful": len(results),
        "churners": sum(result["prediction_value"] for result in results),
        "errors": errors,
        "results": results,
    }


@app.get("/metrics")
def metrics():
    metrics_df = _load_csv(METRICS_PATH)
    metrics_df.columns = [column.strip().lower() for column in metrics_df.columns]
    required = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    missing = sorted(set(required).difference(metrics_df.columns))
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Colonnes absentes dans model_metrics.csv: {', '.join(missing)}",
        )
    return {"models": metrics_df[required].to_dict("records")}


@app.get("/customers/risk")
def customers_at_risk(
    model: str = Query(default="xgboost"),
    limit: int = Query(default=10, ge=1, le=100),
):
    try:
        model_name = _normalize_model_name(model)
        predictions = _all_customer_predictions(model_name)
        columns = [
            "customerID",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "churn_probability",
            "risk_level",
            "customer_segment",
            "recommendation",
        ]
        return {
            "model_used": model_name,
            "customers": predictions[columns].head(limit).to_dict("records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/customers")
def customers(
    model: str = Query(default="xgboost"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=10, le=100),
    search: Optional[str] = Query(default=None),
    contract: Optional[str] = Query(default=None),
    risk_level: Optional[str] = Query(default=None),
):
    try:
        model_name = _normalize_model_name(model)
        predictions = _all_customer_predictions(model_name).copy()
        if search:
            predictions = predictions[
                predictions["customerID"].str.contains(search, case=False, na=False)
            ]
        if contract:
            predictions = predictions[
                predictions["Contract"].str.casefold() == contract.casefold()
            ]
        if risk_level:
            predictions = predictions[
                predictions["risk_level"].str.casefold() == risk_level.casefold()
            ]

        total = int(len(predictions))
        start = (page - 1) * page_size
        selected_columns = [
            "customerID",
            "gender",
            "Contract",
            "InternetService",
            "tenure",
            "MonthlyCharges",
            "Churn",
            "churn_prediction",
            "churn_probability",
            "risk_level",
            "customer_segment",
            "recommendation",
        ]
        records = predictions[selected_columns].iloc[start : start + page_size]
        records = records.where(pd.notna(records), None)
        return {
            "model_used": model_name,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": max((total + page_size - 1) // page_size, 1),
            "customers": records.to_dict("records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/model-details")
def model_details(model: str = Query(default="xgboost")):
    try:
        model_name = _normalize_model_name(model)
        metrics_df = _load_csv(METRICS_PATH)
        display_names = {
            "xgboost": "XGBoost",
            "catboost": "CatBoost",
            "logistic_regression": "Logistic Regression",
        }
        metric = metrics_df[metrics_df["model"] == display_names[model_name]].iloc[0]
        importance_path = MODELS_DIR / f"{display_names[model_name].replace(' ', '_')}_feature_importance.csv"
        importance = _load_csv(importance_path).head(10)
        total_importance = float(importance["importance"].sum()) or 1
        importance["importance_percent"] = (
            importance["importance"] / total_importance * 100
        ).round(2)
        return {
            "model": model_name,
            "metrics": {
                key: round(float(metric[key]), 4)
                for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
            },
            "feature_importance": importance[
                ["feature", "importance", "importance_percent"]
            ].to_dict("records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/analytics")
def analytics(
    contract: Optional[str] = Query(default=None),
    internet_service: Optional[str] = Query(default=None),
    tenure_group: Optional[str] = Query(default=None),
):
    data = _load_csv(DATA_PATH)
    data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce")
    data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce")
    data["is_churn"] = data["Churn"].eq("Yes")
    data["tenure_group"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12 mois", "13-24 mois", "25-48 mois", "49+ mois"],
    )

    filters = {
        "contract": contract,
        "internet_service": internet_service,
        "tenure_group": tenure_group,
    }
    if contract:
        data = data[data["Contract"].str.casefold() == contract.casefold()]
    if internet_service:
        data = data[
            data["InternetService"].str.casefold() == internet_service.casefold()
        ]
    if tenure_group:
        data = data[data["tenure_group"].astype(str).str.casefold() == tenure_group.casefold()]

    total_customers = int(len(data))
    churn_rate = round(float(data["is_churn"].mean() * 100), 2) if total_customers else 0
    monthly_revenue_at_risk = round(
        float(data.loc[data["is_churn"], "MonthlyCharges"].sum()), 2
    )

    return {
        "filters": filters,
        "kpis": {
            "total_customers": total_customers,
            "churned_customers": int(data["is_churn"].sum()),
            "churn_rate": churn_rate,
            "monthly_revenue_at_risk": monthly_revenue_at_risk,
        },
        "churn_by_contract": _percentage_records(
            data.groupby("Contract", observed=True)["is_churn"].mean(), "contract"
        ),
        "churn_by_internet_service": _percentage_records(
            data.groupby("InternetService", observed=True)["is_churn"].mean(),
            "internet_service",
        ),
        "churn_by_tenure_group": _percentage_records(
            data.groupby("tenure_group", observed=True)["is_churn"].mean(),
            "tenure_group",
        ),
    }

