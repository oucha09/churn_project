import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from preprocessing import clean_raw_data, load_and_preprocess, transform_clients


DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def test_clean_raw_data_handles_duplicates_blanks_and_numeric_values():
    raw = pd.read_csv(DATA_PATH)
    dirty = pd.concat([raw.iloc[:2], raw.iloc[:1], raw.iloc[:1]], ignore_index=True)
    dirty.loc[0, "TotalCharges"] = " "
    dirty.loc[1, "PaymentMethod"] = None

    cleaned = clean_raw_data(dirty)

    assert len(cleaned) == 3
    assert cleaned["TotalCharges"].isna().sum() == 1
    assert cleaned["PaymentMethod"].isna().sum() == 1
    assert "customerID" not in cleaned.columns


def test_training_and_inference_use_the_same_features():
    X_train, X_test, _, _, _, _, feature_columns = load_and_preprocess(DATA_PATH)
    raw = pd.read_csv(DATA_PATH).iloc[[0]].drop(columns=["Churn"])
    raw.loc[:, "PaymentMethod"] = "Nouvelle methode"

    transformed = transform_clients(raw)

    assert list(X_train.columns) == feature_columns
    assert list(X_test.columns) == feature_columns
    assert list(transformed.columns) == feature_columns
    assert transformed.isna().sum().sum() == 0


def test_inference_preserves_duplicate_client_rows():
    raw = pd.read_csv(DATA_PATH).drop(columns=["Churn"]).iloc[[0]]
    duplicated = pd.concat([raw, raw], ignore_index=True)

    transformed = transform_clients(duplicated)

    assert len(transformed) == len(duplicated)
