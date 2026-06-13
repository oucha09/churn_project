import pandas as pd
import numpy as np
import joblib

def load_model(model_name="xgboost"):
    """
    Charge un modèle sauvegardé.
    model_name : 'xgboost', 'catboost', ou 'xgboost_optimized'
    """
    model  = joblib.load(f'../models/{model_name}_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    cols   = joblib.load('../models/feature_columns.pkl')
    return model, scaler, cols


def preprocess_single(client_dict, feature_cols):
    """
    Prépare un dictionnaire client (ex: depuis Streamlit)
    en DataFrame prêt pour la prédiction.
    """
    df = pd.DataFrame([client_dict])

    # Ajouter les colonnes manquantes (one-hot non cochées = 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]


def predict_churn(client_dict, model_name="xgboost"):
    """
    Prédit le churn pour un client donné.
    Retourne un dictionnaire avec :
      - prediction  : 0 ou 1
      - probability : float entre 0 et 1
      - risk_level  : 'Faible', 'Moyen', 'Élevé'
      - recommendation : texte d'action
    """
    model, scaler, feature_cols = load_model(model_name)
    input_df = preprocess_single(client_dict, feature_cols)

    prediction  = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    # Niveau de risque
    if probability >= 0.70:
        risk_level     = "Élevé"
        recommendation = ("Contacter le client immédiatement. "
                          "Proposer une offre de rétention : remise sur contrat annuel, "
                          "TechSupport et OnlineSecurity offerts 3 mois.")
    elif probability >= 0.40:
        risk_level     = "Moyen"
        recommendation = ("Envoyer une enquête de satisfaction par email. "
                          "Proposer un programme de fidélité ou réviser la tarification.")
    else:
        risk_level     = "Faible"
        recommendation = ("Client stable. Maintenir la relation standard "
                          "et valoriser sa fidélité.")

    return {
        "prediction"     : prediction,
        "probability"    : round(probability, 4),
        "risk_level"     : risk_level,
        "recommendation" : recommendation,
        "label"          : "CHURN" if prediction == 1 else "NO CHURN"
    }


def predict_batch(df_clients, model_name="xgboost"):
    """
    Prédit le churn pour un DataFrame entier de clients.
    Utile pour des analyses en masse (ex: base CRM complète).
    Retourne le DataFrame original enrichi de 3 colonnes :
      - churn_prediction, churn_probability, risk_level
    """
    model, scaler, feature_cols = load_model(model_name)

    # S'assurer que toutes les colonnes sont présentes
    for col in feature_cols:
        if col not in df_clients.columns:
            df_clients[col] = 0

    X = df_clients[feature_cols]

    df_clients = df_clients.copy()
    df_clients['churn_prediction']  = model.predict(X)
    df_clients['churn_probability'] = model.predict_proba(X)[:, 1].round(4)
    df_clients['risk_level'] = df_clients['churn_probability'].apply(
        lambda p: "Élevé" if p >= 0.70 else ("Moyen" if p >= 0.40 else "Faible")
    )

    return df_clients.sort_values('churn_probability', ascending=False)


# ── Test rapide ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    exemple_client = {
        'gender': 0,
        'SeniorCitizen': 0,
        'Partner': 1,
        'Dependents': 0,
        'tenure': 3,
        'PhoneService': 1,
        'PaperlessBilling': 1,
        'MonthlyCharges': 75.5,
        'TotalCharges': 226.5,
        'InternetService_Fiber optic': 1,
        'InternetService_No': 0,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 1,
        'PaymentMethod_Mailed check': 0,
    }

    result = predict_churn(exemple_client, model_name="xgboost")
    print(f"Résultat  : {result['label']}")
    print(f"Probabilité : {result['probability']:.1%}")
    print(f"Risque    : {result['risk_level']}")
    print(f"Action    : {result['recommendation']}")