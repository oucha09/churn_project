from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
RAW_FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def clean_raw_data(df, drop_duplicates=True):
    """Nettoie les donnees brutes sans apprendre d'information statistique."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.drop(columns=DROP_COLUMNS, errors="ignore")

    for column in cleaned.select_dtypes(include="object").columns:
        stripped = cleaned[column].str.strip()
        cleaned[column] = stripped.where(stripped.notna() & stripped.ne(""), np.nan)

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    return cleaned


def build_preprocessor():
    """Construit le transformateur ajuste uniquement sur les donnees train."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )


def _to_dataframe(values, preprocessor, index):
    return pd.DataFrame(
        values,
        columns=preprocessor.get_feature_names_out(),
        index=index,
    )


def load_preprocessor():
    return joblib.load(MODELS_DIR / "preprocessor.pkl")


def transform_clients(clients, preprocessor=None):
    """Nettoie et transforme des clients bruts avec le pipeline entraine."""
    if isinstance(clients, dict):
        clients = pd.DataFrame([clients])

    cleaned = clean_raw_data(clients, drop_duplicates=False)
    missing_columns = [c for c in RAW_FEATURE_COLUMNS if c not in cleaned.columns]
    for column in missing_columns:
        cleaned[column] = np.nan

    preprocessor = preprocessor or load_preprocessor()
    transformed = preprocessor.transform(cleaned[RAW_FEATURE_COLUMNS])
    return _to_dataframe(transformed, preprocessor, cleaned.index)


def load_and_preprocess(filepath):
    """Charge, separe puis pretraite les donnees sans fuite train/test."""
    df = clean_raw_data(pd.read_csv(filepath))

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Colonne cible absente : {TARGET_COLUMN}")

    invalid_targets = df[TARGET_COLUMN].isna() | ~df[TARGET_COLUMN].isin(["Yes", "No"])
    if invalid_targets.any():
        raise ValueError(f"{invalid_targets.sum()} valeur(s) cible invalides")

    X = df[RAW_FEATURE_COLUMNS]
    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train = _to_dataframe(
        preprocessor.fit_transform(X_train_raw), preprocessor, X_train_raw.index
    )
    X_test = _to_dataframe(
        preprocessor.transform(X_test_raw), preprocessor, X_test_raw.index
    )

    MODELS_DIR.mkdir(exist_ok=True)
    feature_columns = list(preprocessor.get_feature_names_out())
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.pkl")

    # Les deux retours historiques "scaled" sont conserves pour les notebooks.
    return X_train, X_test, y_train, y_test, X_train, X_test, feature_columns
