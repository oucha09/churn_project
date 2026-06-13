import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # ── 1. Nettoyage ───────────────────────────────────
    # Supprimer customerID (inutile)
    df.drop('customerID', axis=1, inplace=True)

    # Corriger TotalCharges (espaces -> NaN -> median)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # ── 2. Encodage de la cible ────────────────────────
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # ── 3. Encodage des variables catégorielles ────────
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling']
    
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    # Label Encoding pour binaires
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # One-Hot Encoding pour multi-catégories
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # ── 4. Séparation X / y ────────────────────────────
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # ── 5. Split Train / Test (80/20) ──────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 6. Normalisation (pour Logistic Regression) ────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Sauvegarder le scaler et les colonnes
    joblib.dump(scaler, '../models/scaler.pkl')
    joblib.dump(list(X.columns), '../models/feature_columns.pkl')

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, list(X.columns)