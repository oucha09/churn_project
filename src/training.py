import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              ConfusionMatrixDisplay, RocCurveDisplay)
import matplotlib.pyplot as plt
import joblib
import numpy as np

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Affiche toutes les métriques + confusion matrix + ROC curve"""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")

    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {model_name}')

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title(f'ROC Curve - {model_name}')
    
    plt.tight_layout()
    plt.savefig(f'../models/{model_name.replace(" ","_")}_eval.png')
    plt.show()

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }


def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=3,   # gère le déséquilibre des classes
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, '../models/xgboost_model.pkl')
    print("✅ XGBoost sauvegardé")
    return model


def train_catboost(X_train, y_train):
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function='Logloss',
        class_weights=[1, 3],   # gère le déséquilibre
        verbose=50,
        random_seed=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, '../models/catboost_model.pkl')
    print("✅ CatBoost sauvegardé")
    return model


def plot_feature_importance(model, feature_names, model_name="XGBoost", top_n=15):
    """Affiche les top N features les plus importantes"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.title(f'Top {top_n} Features Importantes — {model_name}')
    plt.tight_layout()
    plt.savefig(f'../models/{model_name}_feature_importance.png')
    plt.show()