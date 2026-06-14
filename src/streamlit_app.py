import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from prediction import predict_batch
from preprocessing import transform_clients
from retention import customer_segment, retention_recommendation
from training import get_feature_importances
from ui_helpers import get_risk_style


DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_KEYS = {
    "XGBoost": "xgboost",
    "CatBoost": "catboost",
    "Logistic Regression": "logistic_regression",
}
COLORS = {"blue": "#4d8eff", "green": "#34D399", "orange": "#FBBF24", "red": "#F4524A"}


def configure_page(title):
    st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        :root { --bg:#10131a; --card:#191b23; --card2:#272a31; --line:#424754;
                --text:#e1e2ec; --muted:#9ca3af; --primary:#4d8eff; }
        .stApp, [data-testid="stSidebar"] { background:var(--bg); color:var(--text); }
        [data-testid="stSidebar"] { border-right:1px solid rgba(255,255,255,.08); }
        .block-container { max-width:1500px; padding-top:1.5rem; padding-bottom:3rem; }
        h1,h2,h3,p,label,span { color:var(--text); }
        .muted { color:var(--muted); margin-top:-.7rem; margin-bottom:1.3rem; }
        .section-label { color:var(--muted); font-size:.7rem; font-weight:700;
            letter-spacing:.1em; margin:1.2rem 0 .3rem; text-transform:uppercase; }
        .hero-card,.action-card { background:linear-gradient(145deg,rgba(39,42,49,.95),rgba(25,27,35,.95));
            border:1px solid rgba(255,255,255,.1); border-radius:16px; padding:1.4rem 1.6rem; margin-bottom:1rem; }
        .hero-card { display:flex; justify-content:space-between; align-items:center; }
        .eyebrow { color:var(--muted); font-size:.75rem; margin:0 0 .35rem; }
        .result-label { font-size:1.55rem; font-weight:700; margin:0; }
        .probability { font-size:2.6rem; font-weight:750; margin:0; text-align:right; }
        .risk-badge { display:inline-block; border-radius:999px; font-size:.75rem;
            font-weight:700; padding:.3rem .7rem; margin-bottom:.7rem; }
        div[data-testid="stMetric"],div[data-testid="stVerticalBlockBorderWrapper"] {
            background:var(--card); border:1px solid rgba(255,255,255,.1); border-radius:14px; padding:12px; }
        div[data-testid="stMetricValue"] { color:var(--text); }
        button[data-baseweb="tab"] { background:var(--card); border-radius:8px; padding:.6rem 1rem; }
        [data-testid="stDataFrame"] { border:1px solid rgba(255,255,255,.1); border-radius:12px; }
        .stButton > button,.stFormSubmitButton > button { min-height:2.8rem; border-radius:9px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["Churn_bin"] = data["Churn"].eq("Yes").astype(int)
    return data


@st.cache_data
def load_metrics(indexed=False):
    metrics = pd.read_csv(MODELS_DIR / "model_metrics.csv")
    return metrics.set_index("model") if indexed else metrics


@st.cache_resource
def load_models():
    return (
        {
            "XGBoost": joblib.load(MODELS_DIR / "xgboost_model.pkl"),
            "CatBoost": joblib.load(MODELS_DIR / "catboost_model.pkl"),
            "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression_model.pkl"),
        },
        joblib.load(MODELS_DIR / "preprocessor.pkl"),
        joblib.load(MODELS_DIR / "feature_columns.pkl"),
    )


@st.cache_data(show_spinner="Calcul des scores clients...")
def score_customers(model_name):
    scored = predict_batch(load_data().drop(columns=["Churn_bin"]), MODEL_KEYS[model_name])
    scored["Churn_bin"] = scored["Churn"].eq("Yes").astype(int)
    return scored


def page_header(title, subtitle):
    st.title(title)
    st.markdown(f'<p class="muted">{subtitle}</p>', unsafe_allow_html=True)


def style_axes(ax, grid_axis="x"):
    ax.set_facecolor("#191b23")
    ax.figure.set_facecolor("#191b23")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#424754")
    ax.tick_params(colors="#c2c6d6", labelsize=9)
    ax.xaxis.label.set_color("#c2c6d6")
    ax.yaxis.label.set_color("#c2c6d6")
    ax.grid(axis=grid_axis, color="#32353c", linewidth=0.7)
    ax.set_axisbelow(True)


def risk_cell(value):
    style = get_risk_style(value)
    return f"background-color:{style['bg']};color:{style['color']};font-weight:700"


def analytics_filters(key_prefix):
    data = load_data()
    with st.sidebar:
        st.markdown("---")
        st.caption("Filtres de la page")
        model_name = st.selectbox("Modele de risque", list(MODEL_KEYS), key=f"{key_prefix}_model")
        contracts = st.multiselect("Contrat", sorted(data.Contract.unique()), default=sorted(data.Contract.unique()), key=f"{key_prefix}_contract")
        internet = st.multiselect("Internet", sorted(data.InternetService.unique()), default=sorted(data.InternetService.unique()), key=f"{key_prefix}_internet")
        tenure = st.slider("Anciennete", 0, 72, (0, 72), key=f"{key_prefix}_tenure")
    scored = score_customers(model_name)
    filtered = scored[
        scored.Contract.isin(contracts)
        & scored.InternetService.isin(internet)
        & scored.tenure.between(*tenure)
    ].copy()
    return model_name, scored, filtered


def render_dashboard():
    _, _, filtered = analytics_filters("dashboard")
    page_header("Customer Churn Overview", "Vue synthetique du portefeuille et du risque.")
    if filtered.empty:
        st.warning("Aucun client ne correspond aux filtres.")
        return
    actual_churn = filtered.Churn.eq("Yes").mean()
    high_risk = filtered.churn_probability.ge(0.6)
    kpis = st.columns(4)
    kpis[0].metric("Clients", f"{len(filtered):,}")
    kpis[1].metric("Taux de churn", f"{actual_churn:.1%}")
    kpis[2].metric("Revenu mensuel a risque", f"{filtered.loc[high_risk, 'MonthlyCharges'].sum():,.0f}")
    kpis[3].metric("Clients a risque eleve", f"{high_risk.sum():,}")

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Churn par contrat")
        rates = filtered.groupby("Contract")["Churn_bin"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.barh(rates.index, rates.values * 100, color=[get_risk_style(v)["color"] for v in rates])
        ax.bar_label(bars, fmt="%.1f%%", color="#e1e2ec")
        style_axes(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right, st.container(border=True):
        st.subheader("Distribution du risque")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(filtered.churn_probability, bins=25, color=COLORS["blue"], edgecolor="#10131a")
        ax.set_xlabel("Probabilite de churn")
        style_axes(ax, "y")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Charges mensuelles")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(filtered.MonthlyCharges, bins=25, color=COLORS["orange"], edgecolor="#10131a")
        style_axes(ax, "y")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right, st.container(border=True):
        st.subheader("Churn par anciennete")
        groups = pd.cut(filtered.tenure, [-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
        rates = filtered.groupby(groups, observed=True)["Churn_bin"].mean()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(rates.index.astype(str), rates.values * 100, color=[get_risk_style(v)["color"] for v in rates])
        style_axes(ax, "y")
        st.pyplot(fig, width="stretch")
        plt.close(fig)


def render_prediction():
    models, preprocessor, feature_columns = load_models()
    metrics = load_metrics(indexed=True)
    page_header("AI Prediction Command Center", "Prediction du churn et recommandation de fidelisation.")
    left, right = st.columns([1, 2])
    with left, st.container(border=True):
        with st.form("prediction_form"):
            model_name = st.selectbox("Strategie active", list(models))
            gender = st.selectbox("Genre", ["Male", "Female"])
            senior = st.toggle("Senior citizen")
            partner = st.selectbox("Partenaire", ["Yes", "No"])
            dependents = st.selectbox("Dependants", ["No", "Yes"])
            contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
            tenure = st.slider("Anciennete", 0, 72, 12)
            phone = st.selectbox("Telephone", ["Yes", "No"])
            internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
            monthly = st.number_input("Charges mensuelles", 0.0, 200.0, 65.0)
            payment = st.selectbox("Paiement", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            submitted = st.form_submit_button("Predire le risque", type="primary", width="stretch")
    if submitted:
        unavailable = "No internet service" if internet == "No" else "No"
        client = {
            "gender": gender, "SeniorCitizen": int(senior), "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": "No" if phone == "Yes" else "No phone service",
            "InternetService": internet, "OnlineSecurity": unavailable, "OnlineBackup": unavailable,
            "DeviceProtection": unavailable, "TechSupport": unavailable, "StreamingTV": unavailable,
            "StreamingMovies": unavailable, "Contract": contract, "PaperlessBilling": "Yes",
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": monthly * max(tenure, 1),
        }
        transformed = transform_clients(client, preprocessor)
        probability = float(models[model_name].predict_proba(transformed)[0][1])
        st.session_state.prediction_result = {
            "prediction": int(models[model_name].predict(transformed)[0]), "probability": probability,
            "segment": customer_segment(client, probability), "recommendation": retention_recommendation(client, probability),
            "model": model_name,
        }
    result = st.session_state.get("prediction_result")
    active_model = result["model"] if result else "XGBoost"
    with right:
        if result:
            style = get_risk_style(result["probability"])
            label = "Churn probable" if result["prediction"] else "Client probablement fidele"
            st.markdown(
                f"""<div class="hero-card"><div><p class="eyebrow">Prediction</p><p class="result-label" style="color:{style['color']}">{label}</p></div>
                <div><p class="eyebrow">Probabilite</p><p class="probability">{result['probability']:.0%}</p></div></div>
                <div class="action-card"><span class="risk-badge" style="color:{style['color']};background:{style['bg']}">{style['label']}</span>
                <h3>Segment : {result['segment']}</h3><p>{result['recommendation']}</p></div>""", unsafe_allow_html=True)
        else:
            st.info("Saisissez un profil client pour calculer son risque.")
        row = metrics.loc[active_model]
        cols = st.columns(5)
        for col, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"], ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            col.metric(label, f"{row[key]:.3f}")
    importance_tab, roc_tab, comparison_tab = st.tabs(["Feature Importance", "ROC Curve", "Model Comparison"])
    with importance_tab:
        importances = get_feature_importances(models[active_model])
        indices = np.argsort(importances)[-12:]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([feature_columns[i] for i in indices], importances[indices], color=COLORS["blue"])
        style_axes(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with roc_tab:
        st.image(MODELS_DIR / f"{active_model.replace(' ', '_')}_eval.png", width="stretch")
    with comparison_tab:
        st.dataframe(metrics.style.highlight_max(axis=0, color="#234b78").format("{:.3f}"), width="stretch")


def render_customers():
    _, _, filtered = analytics_filters("customers")
    page_header("Customers", "Tous les clients, scores de risque, segments et recommandations.")
    if filtered.empty:
        st.warning("Aucun client ne correspond aux filtres.")
        return
    c1, c2, c3 = st.columns(3)
    search = c1.text_input("Rechercher un identifiant")
    risk_filter = c2.multiselect("Niveau de risque", sorted(filtered.risk_level.unique()), default=sorted(filtered.risk_level.unique()))
    rows = c3.selectbox("Lignes affichees", [25, 50, 100, 250, 500], index=1)
    view = filtered[filtered.risk_level.isin(risk_filter)]
    if search:
        view = view[view.customerID.str.contains(search, case=False, na=False)]
    view = view[["customerID", "gender", "Contract", "InternetService", "tenure", "MonthlyCharges", "Churn",
                 "churn_probability", "risk_level", "customer_segment", "recommendation"]].head(rows)
    st.caption(f"{len(view):,} affiches sur {len(filtered):,} clients filtres")
    st.dataframe(view.style.map(risk_cell, subset=["churn_probability"]).format(
        {"MonthlyCharges": "{:.2f}", "churn_probability": "{:.1%}"}), width="stretch", hide_index=True, height=720)


def render_models():
    models, _, feature_columns = load_models()
    metrics = load_metrics()
    page_header("Model Insights", "Live metrics, performances, backtesting et importance des variables.")
    best = metrics.loc[metrics.roc_auc.idxmax()]
    cols = st.columns(4)
    cols[0].metric("Etat", "Operationnel")
    cols[1].metric("Modeles charges", len(models))
    cols[2].metric("Variables", len(feature_columns))
    cols[3].metric("Meilleur ROC AUC", f"{best['model']} · {best['roc_auc']:.3f}")
    st.subheader("Performance Matrix")
    st.dataframe(metrics.style.highlight_max(axis=0, color="#234b78").format(precision=3), width="stretch", hide_index=True)
    st.bar_chart(metrics.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]], width="stretch")
    st.subheader("Backtesting")
    for _, row in metrics.iterrows():
        with st.container(border=True):
            st.subheader(row["model"])
            metric_cols = st.columns(5)
            for col, key in zip(metric_cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
                col.metric(key.upper(), f"{row[key]:.3f}")
    st.subheader("Feature Importance")
    selected = st.selectbox("Modele analyse", list(models), key="model_importance")
    importances = get_feature_importances(models[selected])
    indices = np.argsort(importances)[-12:]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([feature_columns[i] for i in indices], importances[indices], color=COLORS["blue"])
    style_axes(ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
