# Prediction du Churn Client

Projet de machine learning permettant d'identifier les clients susceptibles de
quitter un service telecom et de proposer des actions de fidelisation.

## Structure du repository

```text
churn_project/
|-- data/
|   `-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|-- notebooks/
|   |-- 01_EDA.ipynb
|   `-- 02_Modelisation.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- training.py
|   |-- prediction.py
|   `-- retention.py
|-- models/
|   |-- xgboost_model.pkl
|   |-- catboost_model.pkl
|   |-- logistic_regression_model.pkl
|   |-- preprocessor.pkl
|   `-- model_metrics.csv
|-- Bonus/
|   |-- dashboard_analytics.py
|   |-- api_fastapi.py
|   `-- hyperparameter_tuning.py
|-- tests/
|-- app.py
|-- train_and_evaluate.py
`-- requirements.txt
```

## Contenu

- `notebooks/` : analyse exploratoire des donnees et experimentations.
- `src/preprocessing.py` : nettoyage, imputation et encodage des donnees.
- `src/training.py` : entrainement et evaluation des modeles.
- `src/prediction.py` : predictions unitaires et batch.
- `src/retention.py` : segmentation des clients et recommandations automatisees.
- `models/` : modeles et preprocesseur sauvegardes au format `.pkl`, metriques,
  graphiques et resultats d'optimisation.
- `app.py` : application Streamlit de prediction et comparaison des modeles.
- `Bonus/dashboard_analytics.py` : dashboard analytique complet.
- `Bonus/hyperparameter_tuning.py` : optimisation GridSearchCV/RandomizedSearchCV.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Entrainement et evaluation

```bash
python train_and_evaluate.py
```

Cette commande entraine XGBoost, CatBoost et Logistic Regression, puis sauvegarde
les modeles, les metriques, les courbes ROC, les matrices de confusion et les
importances de variables dans `models/`.

## Optimisation des hyperparametres

Validation rapide :

```bash
python Bonus/hyperparameter_tuning.py --quick
```

Recherche complete :

```bash
python Bonus/hyperparameter_tuning.py
```

## Applications Streamlit

### Application multipage ChurnPulse

L'application principale `app.py` fournit quatre pages Streamlit distinctes :

- `Dashboard` : KPIs et analyses graphiques ;
- `Prediction` : formulaire client, prediction et recommandation ;
- `Customers` : liste, recherche et filtrage des clients ;
- `Models` : performances, backtesting et importance des variables.

Application multipage :

```bash
python -m streamlit run app.py
```

### Dashboard analytique autonome

Le dashboard `Bonus/dashboard_analytics.py` reste disponible en lancement
autonome et reprend la page Dashboard de l'application multipage.

Dashboard analytique :

```bash
python -m streamlit run Bonus/dashboard_analytics.py
```

## Tests

```bash
python -m pytest -q
```

## Modeles disponibles

- XGBoost
- CatBoost
- Logistic Regression

Les performances comparatives sont disponibles dans
`models/model_metrics.csv`. Les meilleurs hyperparametres identifies sont
enregistres dans `models/best_hyperparameters.json`.
