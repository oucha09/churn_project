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

### Application de prediction

L'application principale `app.py` fournit :

- un formulaire interactif de saisie des informations client ;
- le choix entre XGBoost, CatBoost et Logistic Regression ;
- l'affichage de la prediction `Churn` ou `No Churn` ;
- la probabilite estimee de churn ;
- le segment client et une recommandation de fidelisation automatisee ;
- les matrices de confusion et courbes ROC ;
- la distribution des probabilites, les importances de variables et la
  comparaison des modeles.

Application de prediction :

```bash
python -m streamlit run app.py
```

### Dashboard analytique interactif

Le dashboard `Bonus/dashboard_analytics.py` fournit :

- des filtres interactifs par contrat, service Internet et anciennete ;
- les KPIs de churn et le revenu mensuel a risque ;
- les distributions du churn et des charges mensuelles ;
- les analyses par contrat, service Internet et anciennete ;
- la comparaison des modeles et les resultats d'optimisation ;
- la segmentation des clients a risque ;
- les recommandations de fidelisation automatisees.

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
