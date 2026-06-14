import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.append(os.path.join(BASE_DIR, "src"))

from preprocessing import transform_clients
from ui_helpers import get_risk_style


st.set_page_config(
    page_title="Dashboard Churn Client",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#111111; --muted:#6B7280; --line:#E5E7EB; --surface:#F5F4EE; }
    .stApp, [data-testid="stSidebar"] { background:#FFFFFF; color:var(--ink); }
    .block-container { max-width:1400px; padding-top:1.5rem; padding-bottom:3rem; }
    [data-testid="stSidebar"] { border-right:1px solid var(--line); }
    h1, h2, h3, p, label, span { color:var(--ink); }
    .page-subtitle { color:var(--muted); margin-top:-.6rem; margin-bottom:1.2rem; }
    div[data-testid="stMetric"] {
        background-color:var(--surface); border:1px solid #EBE9E0;
        border-radius:8px; padding:12px 16px;
    }
    div[data-testid="stMetricValue"] { color:var(--ink); font-size:20px; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:#FAFAF8; border:1px solid var(--line); border-radius:10px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        font-size:13px; font-weight:600; margin-bottom:0;
    }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

BLUE = "#4C78A8"
GREEN = "#639922"
ORANGE = "#BA7517"
RED = "#E24B4A"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["Churn_bin"] = (data["Churn"] == "Yes").astype(int)
    return data


@st.cache_resource
def load_models():
    return (
        {
            "XGBoost": joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl")),
            "CatBoost": joblib.load(os.path.join(MODELS_DIR, "catboost_model.pkl")),
            "Logistic Regression": joblib.load(
                os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
            ),
        },
        joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl")),
    )


def style_axes(ax, grid_axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D1D5DB")
    ax.tick_params(colors="#4B5563", labelsize=9)
    ax.grid(axis=grid_axis, color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)


def semantic_color(rate):
    return get_risk_style(rate)["color"]


def risk_cell(value):
    style = get_risk_style(value)
    return f"background-color:{style['bg']}; color:{style['color']}; font-weight:600"


data = load_data()
models, preprocessor = load_models()

with st.sidebar:
    st.caption("Dashboard analytique")

st.title("Dashboard analytique du churn")
st.markdown(
    '<p class="page-subtitle">Vue synthetique des clients, du risque et du revenu expose.</p>',
    unsafe_allow_html=True,
)

filter_1, filter_2, filter_3, filter_4 = st.columns(4)
with filter_1:
    contracts = st.multiselect(
        "Contrat", sorted(data["Contract"].unique()), default=sorted(data["Contract"].unique())
    )
with filter_2:
    internet_services = st.multiselect(
        "Service Internet",
        sorted(data["InternetService"].unique()),
        default=sorted(data["InternetService"].unique()),
    )
with filter_3:
    tenure_range = st.slider(
        "Anciennete (mois)",
        int(data["tenure"].min()),
        int(data["tenure"].max()),
        (int(data["tenure"].min()), int(data["tenure"].max())),
    )
with filter_4:
    model_name = st.selectbox("Modele de risque", list(models))

filtered = data[
    data["Contract"].isin(contracts)
    & data["InternetService"].isin(internet_services)
    & data["tenure"].between(*tenure_range)
].copy()

if filtered.empty:
    st.warning("Aucun client ne correspond aux filtres selectionnes.")
    st.stop()

transformed = transform_clients(filtered, preprocessor)
filtered["churn_probability"] = models[model_name].predict_proba(transformed)[:, 1]

total_clients = len(filtered)
churn_rate = filtered["Churn_bin"].mean()
high_risk = filtered["churn_probability"] >= 0.6
revenue_at_risk = filtered.loc[high_risk, "MonthlyCharges"].sum()

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Clients", f"{total_clients:,}")
kpi_2.metric("Taux de churn", f"{churn_rate:.1%}")
kpi_3.metric("Revenu mensuel a risque", f"{revenue_at_risk:,.0f} MAD")
kpi_4.metric("Clients a risque eleve", f"{high_risk.sum():,}")

row_1_col_1, row_1_col_2 = st.columns(2)
with row_1_col_1:
    with st.container(border=True):
        st.subheader("Repartition du churn")
        churn_percent = churn_rate * 100
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.barh(["Clients"], [100 - churn_percent], color=GREEN, label="Fideles")
        ax.barh(
            ["Clients"],
            [churn_percent],
            left=[100 - churn_percent],
            color=RED,
            label="Churn",
        )
        ax.text((100 - churn_percent) / 2, 0, f"{100 - churn_percent:.1f}%", ha="center", va="center")
        ax.text(100 - churn_percent / 2, 0, f"{churn_percent:.1f}%", ha="center", va="center")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Part des clients (%)")
        ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.45))
        style_axes(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

with row_1_col_2:
    with st.container(border=True):
        st.subheader("Churn par type de contrat")
        contract_rates = filtered.groupby("Contract")["Churn_bin"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 2.8))
        bars = ax.barh(
            contract_rates.index,
            contract_rates.values * 100,
            color=[semantic_color(value) for value in contract_rates.values],
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_xlabel("Taux de churn (%)")
        ax.set_xlim(0, max(contract_rates.max() * 125, 10))
        style_axes(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

row_2_col_1, row_2_col_2 = st.columns(2)
with row_2_col_1:
    with st.container(border=True):
        st.subheader("Distribution des charges mensuelles")
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.hist(filtered["MonthlyCharges"], bins=25, color=BLUE, edgecolor="white")
        ax.set_xlabel("Charges mensuelles")
        ax.set_ylabel("Clients")
        style_axes(ax, grid_axis="y")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

with row_2_col_2:
    with st.container(border=True):
        st.subheader("Churn par anciennete")
        tenure_groups = pd.cut(
            filtered["tenure"],
            bins=[-1, 6, 12, 24, 48, 72],
            labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
        )
        tenure_rates = filtered.groupby(tenure_groups, observed=True)["Churn_bin"].mean()
        fig, ax = plt.subplots(figsize=(6, 2.8))
        bars = ax.bar(
            tenure_rates.index.astype(str),
            tenure_rates.values * 100,
            color=[semantic_color(value) for value in tenure_rates.values],
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_ylabel("Taux de churn (%)")
        ax.set_xlabel("Anciennete (mois)")
        ax.set_ylim(0, max(tenure_rates.max() * 125, 10))
        style_axes(ax, grid_axis="y")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

st.subheader("Clients a risque eleve")
high_risk_clients = (
    filtered.loc[
        high_risk,
        ["customerID", "Contract", "tenure", "MonthlyCharges", "churn_probability"],
    ]
    .sort_values("churn_probability", ascending=False)
    .head(100)
    .rename(
        columns={
            "customerID": "Client",
            "Contract": "Contrat",
            "tenure": "Anciennete",
            "MonthlyCharges": "Charges mensuelles",
            "churn_probability": "Score de risque",
        }
    )
)

st.dataframe(
    high_risk_clients.style.map(risk_cell, subset=["Score de risque"]).format(
        {"Charges mensuelles": "{:.2f}", "Score de risque": "{:.1%}"}
    ),
    width="stretch",
    hide_index=True,
)
