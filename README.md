# GetAround - Analyse des retards & Pricing API

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

> **Bloc 5 - Jedha Bootcamp** | Industrialisation d'un algorithme d'apprentissage automatique et automatisation des processus de decision

---

## Contexte du projet

GetAround est une plateforme de location de voitures entre particuliers. Les retards au checkout generent des frictions pour les conducteurs suivants (attente, voire annulation). L'objectif est de definir un **seuil minimum entre deux locations** consecutives pour reduire les conflits, tout en minimisant l'impact sur le chiffre d'affaires.

### Livrables

| Livrable | Description | Lien |
|---|---|---|
| **Dashboard Streamlit** | Analyse interactive des retards et simulation de seuils | [Acceder au dashboard](https://athanormark-getaround-dashboard.hf.space) |
| **API Pricing (FastAPI)** | Endpoint `/predict` pour la prediction du prix journalier | [Acceder a l'API](https://athanormark-getaround-pricing-api.hf.space) |
| **Documentation API** | Page `/docs` avec description des endpoints | [Voir la documentation](https://athanormark-getaround-pricing-api.hf.space/docs) |
| **Swagger interactif** | Interface de test `/swagger` (OpenAPI) | [Tester l'API](https://athanormark-getaround-pricing-api.hf.space/swagger) |

---

## Partie 1 - Analyse des retards (EDA)

### Donnees

- **Source** : [get_around_delay_analysis.xlsx](https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx)
- **Volume** : 21 310 locations, 7 colonnes
- **Types de checkin** : Mobile (80%) et Connect (20%)

### Chiffres cles

| Indicateur | Valeur |
|---|---|
| Locations terminees | 18 045 (84.7%) |
| Locations annulees | 3 265 (15.3%) |
| En retard au checkout | 57.5% des locations avec donnees de checkout |
| Retard moyen (quand retard) | 202 min (3h22) |
| Retard median (quand retard) | 53 min |
| Locations consecutives | 1 841 (8.6%) |
| Cas problematiques (retard > buffer) | 218 (11.8% des consecutives) |
| Annulations liees aux retards | 37 (17% des cas problematiques) |

**Observation notable** : les locations **Connect** sont en moyenne rendues en avance (median -9 min), tandis que les locations **Mobile** sont en moyenne rendues en retard (median +14 min).

### Simulation des seuils

![Tradeoff seuil](assets/images/threshold_tradeoff.png)

### Recommandation

**Seuil de 120 minutes, scope Connect uniquement**

- Resout **84%** des cas problematiques pour les voitures Connect
- Ne bloque que **36%** des locations consecutives Connect
- Impact revenus limite : les voitures Connect representent 20% du parc
- Le checkin sans contact des voitures Connect les rend plus sensibles aux retards

---

## Partie 2 - Pricing ML

### Donnees

- **Source** : [get_around_pricing_project.csv](https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv)
- **Volume** : 4 843 vehicules, 13 features, aucune valeur manquante
- **Target** : `rental_price_per_day` (prix journalier en EUR)

### Pipeline

1. **Preprocessing** (`ColumnTransformer`) : `StandardScaler` (numeriques), `OneHotEncoder` avec `handle_unknown='ignore'` (categoriques), passthrough (booleens)
2. **Modeles** : Linear Regression (baseline) et Gradient Boosting (200 estimators, depth 5, lr 0.1)
3. **Validation** : train/test split 80/20 + cross-validation 5 folds
4. **Tracking** : MLflow (parametres, metriques, artefacts)

### Resultats

| Modele | R2 | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.6937 | 12.12 EUR | 17.96 EUR |
| **Gradient Boosting** | **0.7504** | **10.29 EUR** | **16.22 EUR** |

Cross-validation Gradient Boosting : R2 = 0.693 (+/- 0.070) — leger overfitting par rapport au test set (0.75), ecart modere et performances stables.

### Feature importance

![Feature Importance](assets/images/feature_importance.png)

Features les plus correlees au prix : puissance moteur (+0.63), kilometrage (-0.45), automatique (+0.42), Connect (+0.32), GPS (+0.31). Les pneus hiver n'ont aucun impact (+0.02).

### Predictions vs valeurs reelles

![Predictions vs Reel](assets/images/predictions_vs_actual.png)

---

## Tester l'API

### Avec curl

```bash
curl -X POST "https://athanormark-getaround-pricing-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"input": [["Renault", 140000, 135, "diesel", "black", "sedan", true, true, false, false, true, true, true]]}'
```

### Avec Python

```python
import requests

response = requests.post("https://athanormark-getaround-pricing-api.hf.space/predict", json={
    "input": [["Renault", 140000, 135, "diesel", "black", "sedan", True, True, False, False, True, True, True]]
})
print(response.json())
# {"prediction": [139.12]}
```

---

## Installation locale

### Prerequis

- Python 3.10+
- Docker (optionnel)

### Setup

```bash
git clone https://github.com/athanormark/GETAROUND-BLOC-5_JEDHA_FORMATION.git
cd GETAROUND-BLOC-5_JEDHA_FORMATION
pip install -r requirements.txt
```

Les datasets sont a telecharger dans `data/` :
- [Delay Analysis (.xlsx)](https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx)
- [Pricing (.csv)](https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv)

```bash
# Notebook
jupyter notebook getaround_analysis.ipynb

# Dashboard
cd dashboard && streamlit run app.py

# API
cd api && uvicorn app:app --reload
```

### Docker

```bash
# API
cd api && docker build -t getaround-api . && docker run -p 7860:7860 getaround-api

# Dashboard
cd dashboard && docker build -t getaround-dashboard . && docker run -p 7860:7860 getaround-dashboard
```

---

## Structure du projet

```
GETAROUND-BLOC-5_JEDHA_FORMATION/
├── getaround_analysis.ipynb       # Notebook principal (EDA + ML + MLflow)
├── dashboard/
│   ├── app.py                     # Dashboard Streamlit (theme dark)
│   ├── Dockerfile
│   └── requirements.txt
├── api/
│   ├── app.py                     # API FastAPI (/predict, /docs, /swagger)
│   ├── model.joblib               # Pipeline sklearn (Gradient Boosting)
│   ├── Dockerfile
│   └── requirements.txt
├── assets/images/                 # Graphiques exportes (PNG)
├── data/                          # Datasets (non versionnes, .gitignore)
├── requirements.txt               # Dependances globales
└── README.md
```

---

## Hebergement

Les services sont heberges sur **HuggingFace Spaces** (tier gratuit). Les Spaces peuvent se mettre en veille apres une periode d'inactivite : le premier chargement prend alors quelques secondes le temps du reveil automatique.

---

## Auteur

**Athanor SAVOUILLAN** - Jedha Data Fullstack Bootcamp
