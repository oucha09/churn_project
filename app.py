import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import sys
sys.path.append('./src')

# ── Chargement des modèles ─────────────────────────────
@st.cache_resource
def load_models():
    xgb_model     = joblib.load('models/xgboost_model.pkl')
    cat_model     = joblib.load('models/catboost_model.pkl')
    scaler        = joblib.load('models/scaler.pkl')
    feature_cols  = joblib.load('models/feature_columns.pkl')
    return xgb_model, cat_model, scaler, feature_cols

xgb_model, cat_model, scaler, feature_cols = load_models()

# ── Interface ──────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Système de Prédiction du Churn Client")
st.markdown("*Identifiez les clients à risque de départ*")

# ── Sidebar : Choix du modèle ──────────────────────────
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox("Choisir le modèle", ["XGBoost", "CatBoost"])
model = xgb_model if model_choice == "XGBoost" else cat_model

# ── Formulaire client ──────────────────────────────────
st.header("📝 Informations du Client")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Démographie")
    gender         = st.selectbox("Genre", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner        = st.selectbox("Partenaire", ["Yes", "No"])
    dependents     = st.selectbox("Personnes à charge", ["Yes", "No"])

with col2:
    st.subheader("Abonnement")
    tenure         = st.slider("Durée abonnement (mois)", 0, 72, 12)
    contract       = st.selectbox("Type de contrat", 
                                   ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Facturation sans papier", ["Yes", "No"])
    payment        = st.selectbox("Méthode de paiement", 
                                   ["Electronic check", "Mailed check", 
                                    "Bank transfer (automatic)", 
                                    "Credit card (automatic)"])

with col3:
    st.subheader("Services & Finances")
    phone_service  = st.selectbox("Service téléphonique", ["Yes", "No"])
    internet       = st.selectbox("Service Internet", 
                                   ["DSL", "Fiber optic", "No"])
    monthly        = st.number_input("Frais mensuels ($)", 0.0, 200.0, 65.0)
    total          = st.number_input("Frais totaux ($)", 0.0, 10000.0, 
                                      float(monthly * tenure))

# ── Prédiction ─────────────────────────────────────────
def build_input(feature_cols):
    """Construit le vecteur d'entrée dans le bon format"""
    # Valeurs de base
    raw = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': senior_citizen,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
    }

    # One-hot pour InternetService
    for val in ['Fiber optic', 'No']:
        raw[f'InternetService_{val}'] = 1 if internet == val else 0

    # One-hot pour Contract
    for val in ['One year', 'Two year']:
        raw[f'Contract_{val}'] = 1 if contract == val else 0

    # One-hot pour PaymentMethod
    for val in ['Credit card (automatic)', 
                'Electronic check', 'Mailed check']:
        raw[f'PaymentMethod_{val}'] = 1 if payment == val else 0

    # Construire le DataFrame avec toutes les colonnes
    input_df = pd.DataFrame([raw])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[feature_cols]

if st.button("🔮 Prédire le Churn", type="primary"):
    input_df = build_input(feature_cols)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.header("📈 Résultat de la Prédiction")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if prediction == 1:
            st.error(f"⚠️ **CHURN PROBABLE** — Ce client risque de partir !")
        else:
            st.success(f"✅ **PAS DE CHURN** — Ce client est fidèle.")

    with col_res2:
        st.metric("Probabilité de Churn", f"{probability:.1%}")
        st.progress(float(probability))

    # Jauge visuelle
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh(0, probability, color='red' if probability > 0.5 else 'green', height=0.5)
    ax.barh(0, 1, color='lightgray', height=0.5, zorder=0)
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2)
    ax.set_yticks([])
    ax.set_xlabel("Probabilité de Churn")
    ax.set_title("Niveau de Risque")
    st.pyplot(fig)

    # Recommandation
    st.header("💡 Recommandation Métier")
    if probability > 0.7:
        st.warning("🔴 **Risque ÉLEVÉ** : Contacter immédiatement le client. "
                   "Proposer une offre de rétention (réduction, upgrade, appel personnalisé).")
    elif probability > 0.4:
        st.info("🟡 **Risque MOYEN** : Surveiller ce client. "
                "Envoyer une communication proactive (email, enquête de satisfaction).")
    else:
        st.success("🟢 **Risque FAIBLE** : Client stable. Continuer à maintenir la relation.")

# ── Visualisations dans l'onglet Stats ────────────────
st.header("📊 Visualisations du Modèle")
tab1, tab2 = st.tabs(["Feature Importance", "Info Modèle"])

with tab1:
    import numpy as np
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-10:]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh([feature_cols[i] for i in top_idx], importances[top_idx], color='steelblue')
    ax2.set_title(f"Top 10 Features — {model_choice}")
    st.pyplot(fig2)

with tab2:
    st.info(f"Modèle actif : **{model_choice}**")
    st.write("Dataset : Telco Customer Churn (IBM/Kaggle)")
    st.write("Split : 80% Train / 20% Test | Stratified")