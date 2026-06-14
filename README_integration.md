# Integration FastAPI et frontend

## Demarrer le backend

Depuis la racine du projet :

```bash
python -m uvicorn Bonus.api_fastapi:app --reload --port 8000
```

La documentation interactive est disponible sur
`http://localhost:8000/docs`.

Pour autoriser un frontend de production, definir son origine avant le
demarrage :

```powershell
$env:FRONTEND_ORIGIN="https://frontend.example.com"
python -m uvicorn Bonus.api_fastapi:app --reload --port 8000
```

Les frontends locaux servis depuis `localhost` ou `127.0.0.1`, quel que soit
leur port, sont autorises par CORS.

## Demarrer les interfaces Streamlit

Application de prediction :

```bash
py -m streamlit run app.py
```

Dashboard analytique :

```bash
py -m streamlit run Bonus/dashboard_analytics.py
```

## Contrat API

### `GET /health`

Retourne l'etat de l'API et la liste des modeles `.pkl` disponibles.

```json
{"status": "ok", "models_loaded": ["xgboost", "catboost", "logistic_regression"]}
```

### `POST /predict?model=xgboost`

Modeles acceptes : `xgboost`, `catboost`, `logistic_regression`.

Le corps JSON accepte les champs :
`gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`,
`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`,
`Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges` et
`TotalCharges`.

Exemple de reponse :

```json
{
  "model_used": "xgboost",
  "prediction": "Churn",
  "probability": 0.78,
  "segment": "Forte valeur a risque",
  "retention_recommendation": "Appel prioritaire sous 24h..."
}
```

### `POST /predict/batch?model=catboost`

Accepte un tableau JSON de clients. La reponse contient `results`, les erreurs
eventuelles par index, le nombre total et le nombre de churners predits.

### `GET /metrics`

Retourne sous `models` les metriques `accuracy`, `precision`, `recall`, `f1` et
`roc_auc` des trois modeles depuis `models/model_metrics.csv`.

### `GET /analytics`

Retourne :

- `kpis` : nombre de clients, clients churnes, taux de churn et revenu mensuel
  a risque ;
- `churn_by_contract` ;
- `churn_by_internet_service` ;
- `churn_by_tenure_group`.

Les filtres optionnels `contract`, `internet_service` et `tenure_group` peuvent
etre passes dans l'URL.

### `GET /customers/risk?model=xgboost&limit=10`

Retourne les clients classes par probabilite de churn, avec leur niveau de
risque, segment et recommandation.

### `GET /customers?page=1&page_size=25`

Retourne tous les clients avec pagination et scores de risque. Les filtres
optionnels sont `search`, `contract`, `risk_level` et `model`.

### `GET /model-details?model=xgboost`

Retourne les metriques et les principales variables influentes du modele pour
les onglets d'analyse.

## Exemple frontend

```js
// MODIFIED: wired to POST /predict
const response = await fetch(
  `http://localhost:8000/predict?model=${selectedModel}`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(customer),
  },
);

if (!response.ok) {
  throw new Error((await response.json()).detail ?? "Erreur API");
}

const prediction = await response.json();
```
