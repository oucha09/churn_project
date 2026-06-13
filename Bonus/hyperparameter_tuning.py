import sys
import os

# ── Chemins absolus ────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
sys.path.append(os.path.join(BASE_DIR, 'src'))

import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from preprocessing import load_and_preprocess
import joblib
import pandas as pd

# ── Chargement des données ─────────────────────────────────────
X_train, X_test, y_train, y_test, _, _, feature_cols = \
    load_and_preprocess(DATA_PATH)

# ══════════════════════════════════════════════════════════════
# OPTIMISATION XGBOOST
# ══════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("  Optimisation XGBoost en cours...")
print("="*50)

param_grid_xgb = {
    'n_estimators'    : [100, 200, 300],
    'max_depth'       : [3, 5, 7],
    'learning_rate'   : [0.01, 0.05, 0.1],
    'subsample'       : [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}

xgb_base = xgb.XGBClassifier(
    scale_pos_weight=3,
    eval_metric='logloss',
    random_state=42
)

search_xgb = RandomizedSearchCV(
    xgb_base, param_grid_xgb,
    n_iter=30, cv=5, scoring='roc_auc',
    n_jobs=-1, random_state=42, verbose=1
)
search_xgb.fit(X_train, y_train)

print("\nMeilleurs params XGBoost :", search_xgb.best_params_)
print("Meilleur ROC-AUC          :", round(search_xgb.best_score_, 4))

joblib.dump(
    search_xgb.best_estimator_,
    os.path.join(MODELS_DIR, 'xgboost_optimized_model.pkl')
)
print("✅ XGBoost optimisé sauvegardé")

# ══════════════════════════════════════════════════════════════
# OPTIMISATION CATBOOST
# ══════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("  Optimisation CatBoost en cours...")
print("="*50)

param_grid_cat = {
    'iterations'   : [200, 300, 500],
    'depth'        : [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg'  : [1, 3, 5, 7],
}

cat_base = CatBoostClassifier(
    loss_function='Logloss',
    class_weights=[1, 3],
    verbose=0,
    random_seed=42
)

search_cat = RandomizedSearchCV(
    cat_base, param_grid_cat,
    n_iter=20, cv=5, scoring='roc_auc',
    n_jobs=-1, random_state=42, verbose=1
)
search_cat.fit(X_train, y_train)

print("\nMeilleurs params CatBoost :", search_cat.best_params_)
print("Meilleur ROC-AUC           :", round(search_cat.best_score_, 4))

joblib.dump(
    search_cat.best_estimator_,
    os.path.join(MODELS_DIR, 'catboost_optimized_model.pkl')
)
print("✅ CatBoost optimisé sauvegardé")

# ══════════════════════════════════════════════════════════════
# TABLEAU COMPARATIF FINAL
# ══════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("  Comparaison des meilleurs résultats")
print("="*50)

results_xgb = pd.DataFrame(search_xgb.cv_results_).sort_values('rank_test_score')
results_cat = pd.DataFrame(search_cat.cv_results_).sort_values('rank_test_score')

print("\nTop 5 XGBoost :")
print(results_xgb[['params', 'mean_test_score', 'std_test_score']].head(5).to_string())

print("\nTop 5 CatBoost :")
print(results_cat[['params', 'mean_test_score', 'std_test_score']].head(5).to_string())

print("\n✅ Optimisation terminée ! Modèles sauvegardés dans /models")