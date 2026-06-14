import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


sys.path.append("./src")
from preprocessing import transform_clients
from retention import customer_segment, retention_recommendation
from training import get_feature_importances
from ui_helpers import get_risk_style, result_card_html, risk_badge_html


st.set_page_config(
    page_title="Prediction du Churn Client",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#111111; --muted:#6B7280; --line:#E5E7EB; --surface:#F5F4EE; }
    .stApp, [data-testid="stSidebar"] { background:#FFFFFF; color:var(--ink); }
    .block-container { max-width:1200px; padding-top:1.5rem; padding-bottom:3rem; }
    [data-testid="stSidebar"] { border-right:1px solid var(--line); }
    [data-testid="stSidebar"] .block-container { padding-top:1.25rem; }
    h1, h2, h3, p, label, span { color:var(--ink); }
    .page-subtitle { color:var(--muted); margin-top:-.6rem; margin-bottom:1.5rem; }
    .group-label {
        color:var(--muted); font-size:.72rem; font-weight:600;
        letter-spacing:.09em; margin:1.35rem 0 .35rem; text-transform:uppercase;
    }
    .result-card, .recommendation-card {
        background:#FFFFFF; border:1px solid var(--line); border-radius:12px;
        padding:18px 22px; margin-bottom:1rem;
    }
    .result-card { display:flex; justify-content:space-between; align-items:center; }
    .eyebrow { color:var(--muted); font-size:.75rem; margin:0 0 .25rem; }
    .prediction-label { font-size:1.35rem; font-weight:600; margin:0; }
    .probability { text-align:right; }
    .probability-value { color:var(--ink); font-size:2rem; font-weight:600; margin:0; }
    .risk-badge {
        display:inline-block; border-radius:999px; font-size:.75rem;
        font-weight:600; padding:.25rem .65rem; margin-bottom:.7rem;
    }
    .recommendation-title { font-size:1rem; font-weight:600; margin:0 0 .35rem; }
    .recommendation-text { color:#374151; margin:0; }
    div[data-testid="stMetric"] {
        background-color:var(--surface); border:1px solid #EBE9E0;
        border-radius:8px; padding:12px 16px;
    }
    div[data-testid="stMetricValue"] { color:var(--ink); font-size:20px; }
    div[data-baseweb="tab-list"] { gap:.35rem; }
    button[data-baseweb="tab"] { padding:.55rem .9rem; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:8px; }
    .stButton > button { min-height:2.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    return (
        {
            "XGBoost": joblib.load("models/xgboost_model.pkl"),
            "CatBoost": joblib.load("models/catboost_model.pkl"),
            "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
        },
        joblib.load("models/preprocessor.pkl"),
        joblib.load("models/feature_columns.pkl"),
    )


@st.cache_data
def load_metrics():
    return pd.read_csv("models/model_metrics.csv", index_col="model")


def section_label(label):
    st.markdown(f'<div class="group-label">{label}</div>', unsafe_allow_html=True)


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D1D5DB")
    ax.tick_params(colors="#4B5563")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)


models, preprocessor, feature_columns = load_models()
metrics = load_metrics()

with st.sidebar:
    st.subheader("Informations client")
    with st.form("client_form"):
        model_name = st.selectbox("Modele", list(models))

        section_label("Profil")
        gender = st.selectbox("Genre", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partenaire", ["Yes", "No"])
        dependents = st.selectbox("Dependants", ["Yes", "No"])

        section_label("Abonnement")
        contract = st.selectbox(
            "Type de contrat", ["Month-to-month", "One year", "Two year"]
        )
        tenure = st.slider("Anciennete (mois)", 0, 72, 12)
        phone_service = st.selectbox("Service telephonique", ["Yes", "No"])
        internet = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
        paperless = st.selectbox("Facturation sans papier", ["Yes", "No"])

        section_label("Paiement")
        monthly = st.number_input("Charges mensuelles", 0.0, 200.0, 65.0)
        payment = st.selectbox(
            "Methode de paiement",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        section_label("")
        submitted = st.form_submit_button(
            "Predire", type="primary", width="stretch"
        )

if submitted:
    unavailable = "No internet service" if internet == "No" else "No"
    client = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": "No" if phone_service == "Yes" else "No phone service",
        "InternetService": internet,
        "OnlineSecurity": unavailable,
        "OnlineBackup": unavailable,
        "DeviceProtection": unavailable,
        "TechSupport": unavailable,
        "StreamingTV": unavailable,
        "StreamingMovies": unavailable,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": monthly * tenure,
    }
    transformed = transform_clients(client, preprocessor)
    model = models[model_name]
    probability = float(model.predict_proba(transformed)[0][1])
    prediction = int(model.predict(transformed)[0])
    st.session_state["prediction_result"] = {
        "prediction": prediction,
        "probability": probability,
        "segment": customer_segment(client, probability),
        "recommendation": retention_recommendation(client, probability),
        "model": model_name,
    }

st.title("Prediction du churn client")
st.markdown(
    '<p class="page-subtitle">Estimation du risque et action de fidelisation recommandee.</p>',
    unsafe_allow_html=True,
)

result = st.session_state.get("prediction_result")
if result:
    probability = result["probability"]
    prediction_label = "Churn probable" if result["prediction"] else "No Churn"
    st.markdown(
        result_card_html(prediction_label, probability), unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="recommendation-card">
            {risk_badge_html(probability)}
            <p class="recommendation-title">Segment : {result["segment"]}</p>
            <p class="recommendation-text">{result["recommendation"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_model = result["model"]
else:
    active_model = "XGBoost"
    neutral = get_risk_style(0.0)
    st.markdown(
        f"""
        <div class="result-card">
            <div>
                <p class="eyebrow">Prediction</p>
                <p class="prediction-label" style="color:{neutral["color"]};">
                    En attente d'une saisie
                </p>
            </div>
            <div class="probability">
                <p class="eyebrow">Probabilite de churn</p>
                <p class="probability-value">--</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

model_metrics = metrics.loc[active_model]
metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Accuracy", f"{model_metrics['accuracy']:.1%}")
metric_2.metric("ROC AUC", f"{model_metrics['roc_auc']:.3f}")
metric_3.metric("F1-score", f"{model_metrics['f1']:.3f}")

st.markdown("### Analyse du modele")
importance_tab, roc_tab, comparison_tab = st.tabs(
    ["Importance des variables", "Courbe ROC", "Comparaison modeles"]
)

with importance_tab:
    model = models[active_model]
    importances = get_feature_importances(model)
    indices = np.argsort(importances)[-12:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [feature_columns[index] for index in indices],
        importances[indices],
        color="#4C78A8",
    )
    ax.set_xlabel("Importance")
    style_axes(ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with roc_tab:
    st.image(
        f"models/{active_model.replace(' ', '_')}_eval.png",
        caption=f"Matrice de confusion et courbe ROC - {active_model}",
        width="stretch",
    )

with comparison_tab:
    st.dataframe(
        metrics.style.highlight_max(axis=0, color="#EAF3DE").format("{:.3f}"),
        width="stretch",
    )
