import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="GetAround Pricing API",
    description="API de prediction du prix journalier de location de voiture",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url=None
)

# Chargement du modele entraine (pipeline sklearn)
model = joblib.load("model.joblib")

# Colonnes attendues par le modele (13 features)
FEATURE_COLUMNS = [
    "model_key", "mileage", "engine_power", "fuel", "paint_color",
    "car_type", "private_parking_available", "has_gps",
    "has_air_conditioning", "automatic_car", "has_getaround_connect",
    "has_speed_regulator", "winter_tires"
]


class PredictionInput(BaseModel):
    """Schema d'un vehicule individuel (non utilise directement)."""

    model_key: str
    mileage: int
    engine_power: int
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool


class PredictionRequest(BaseModel):
    """Corps de la requete : liste de listes (1 sous-liste = 1 vehicule)."""

    input: List[List]


class PredictionResponse(BaseModel):
    """Reponse : liste des prix predits en EUR/jour."""

    prediction: List[float]


# Style CSS commun entre la page d'accueil et la documentation
COMMON_STYLE = """
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: #0f1117; color: #e0e0e0; line-height: 1.7;
    }
    .header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 48px 24px; text-align: center;
        border-bottom: 3px solid #5ce1e6;
    }
    .header h1 { font-size: 2.2rem; font-weight: 700; color: #fff; margin-bottom: 8px; }
    .header .subtitle { color: #8899aa; font-size: 1.1rem; }
    .header .badge {
        display: inline-block; background: #5ce1e6; color: #0f1117;
        padding: 4px 14px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin-top: 12px;
    }
    .container { max-width: 960px; margin: 0 auto; padding: 32px 24px; }
    .card {
        background: #1a1d29; border: 1px solid #2a2d3a;
        border-radius: 12px; padding: 28px; margin-bottom: 24px;
        transition: border-color 0.2s;
    }
    .card:hover { border-color: #5ce1e6; }
    .card h2 {
        color: #5ce1e6; font-size: 1.3rem; margin-bottom: 12px;
        display: flex; align-items: center; gap: 10px;
    }
    .card p { color: #aab; }
    a { color: #5ce1e6; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .btn {
        display: inline-block; padding: 12px 28px; border-radius: 8px;
        font-weight: 600; font-size: 1rem; transition: all 0.2s;
        text-decoration: none !important;
    }
    .btn-primary { background: #5ce1e6; color: #0f1117; }
    .btn-primary:hover { background: #4dc8cd; transform: translateY(-1px); }
    .btn-outline { border: 1px solid #5ce1e6; color: #5ce1e6; background: transparent; }
    .btn-outline:hover { background: rgba(92,225,230,0.1); }
    .metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px; margin: 20px 0;
    }
    .metric {
        background: #12141d; border: 1px solid #2a2d3a;
        border-radius: 10px; padding: 20px; text-align: center;
    }
    .metric .value { font-size: 1.8rem; font-weight: 700; color: #5ce1e6; }
    .metric .label { font-size: 0.85rem; color: #778; margin-top: 4px; }
    code {
        background: #12141d; color: #5ce1e6; padding: 3px 8px;
        border-radius: 4px;
        font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.9em;
    }
    pre {
        background: #12141d; border: 1px solid #2a2d3a;
        border-radius: 8px; padding: 20px; overflow-x: auto;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 0.88rem; color: #c8d0d8; line-height: 1.5;
    }
    table {
        width: 100%; border-collapse: separate; border-spacing: 0;
        border-radius: 8px; overflow: hidden;
        border: 1px solid #2a2d3a; margin: 16px 0;
    }
    th {
        background: #1e2130; color: #5ce1e6; padding: 12px 16px;
        text-align: left; font-weight: 600; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    td { padding: 10px 16px; border-top: 1px solid #2a2d3a; font-size: 0.92rem; }
    tr:hover td { background: rgba(92,225,230,0.03); }
    .method-badge {
        display: inline-block; padding: 4px 12px; border-radius: 4px;
        font-weight: 700; font-size: 0.8rem; font-family: monospace;
    }
    .post { background: rgba(73,204,144,0.15); color: #49cc91; }
    .get { background: rgba(92,225,230,0.15); color: #5ce1e6; }
    .endpoint { font-family: monospace; font-size: 1rem; color: #fff; margin-left: 8px; }
    .nav { display: flex; gap: 12px; justify-content: center; margin-top: 20px; }
    .footer {
        text-align: center; padding: 32px; color: #556;
        font-size: 0.85rem; border-top: 1px solid #1e2130; margin-top: 40px;
    }
</style>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    """Page d'accueil de l'API avec metriques et liens rapides."""
    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>GetAround Pricing API</title>
        {COMMON_STYLE}
    </head>
    <body>
        <div class="header">
            <h1>GetAround Pricing API</h1>
            <p class="subtitle">Prediction du prix journalier de location de voiture par Machine Learning</p>
            <span class="badge">v1.0.0 &mdash; Gradient Boosting</span>
            <div class="nav">
                <a href="/docs" class="btn btn-primary">Documentation</a>
                <a href="/swagger" class="btn btn-outline">Swagger UI</a>
            </div>
        </div>
        <div class="container">
            <div class="metrics">
                <div class="metric"><div class="value">0.75</div><div class="label">R2 Score</div></div>
                <div class="metric"><div class="value">10.29</div><div class="label">MAE (EUR)</div></div>
                <div class="metric"><div class="value">16.22</div><div class="label">RMSE (EUR)</div></div>
                <div class="metric"><div class="value">4 843</div><div class="label">Vehicules entraines</div></div>
            </div>

            <div class="card">
                <h2><span class="method-badge post">POST</span> <span class="endpoint">/predict</span></h2>
                <p>Envoie les caracteristiques d'un vehicule et recois le prix journalier predit.</p>
                <pre>curl -X POST "/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{"input": [["Renault", 140000, 135, "diesel", "black", "sedan", true, true, false, false, true, true, true]]}}'

# Reponse : {{"prediction": [139.12]}}</pre>
            </div>

            <div class="card">
                <h2><span class="method-badge get">GET</span> <span class="endpoint">/docs</span></h2>
                <p>Documentation complete de l'API avec les features attendues et des exemples.</p>
            </div>

            <div class="card">
                <h2><span class="method-badge get">GET</span> <span class="endpoint">/swagger</span></h2>
                <p>Interface Swagger interactive pour tester l'API directement dans le navigateur.</p>
            </div>

            <div class="card">
                <h2>Top 3 features</h2>
                <div class="metrics">
                    <div class="metric"><div class="value">46%</div><div class="label">Puissance moteur</div></div>
                    <div class="metric"><div class="value">27%</div><div class="label">Kilometrage</div></div>
                    <div class="metric"><div class="value">5%</div><div class="label">GPS</div></div>
                </div>
            </div>
        </div>
        <div class="footer">GetAround Pricing API &mdash; Athanor Savouillan &mdash; Jedha Bootcamp</div>
    </body>
    </html>
    """


@app.get("/docs", response_class=HTMLResponse)
def documentation():
    """Page de documentation avec exemples curl et Python."""
    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>GetAround API - Documentation</title>
        {COMMON_STYLE}
    </head>
    <body>
        <div class="header">
            <h1>Documentation API</h1>
            <p class="subtitle">Tout ce qu'il faut pour utiliser l'endpoint de prediction</p>
            <div class="nav">
                <a href="/" class="btn btn-outline">Accueil</a>
                <a href="/swagger" class="btn btn-primary">Tester avec Swagger</a>
            </div>
        </div>
        <div class="container">

            <div class="card">
                <h2><span class="method-badge post">POST</span> <span class="endpoint">/predict</span></h2>
                <p>Predit le prix journalier de location a partir des caracteristiques du vehicule.</p>
            </div>

            <div class="card">
                <h2>Features attendues</h2>
                <p>JSON avec une cle <code>input</code> contenant une liste de listes. Chaque sous-liste = 1 vehicule (13 features) :</p>
                <table>
                    <tr><th>#</th><th>Feature</th><th>Type</th><th>Exemple</th></tr>
                    <tr><td>0</td><td>model_key</td><td><code>string</code></td><td>Renault</td></tr>
                    <tr><td>1</td><td>mileage</td><td><code>int</code></td><td>140 000</td></tr>
                    <tr><td>2</td><td>engine_power</td><td><code>int</code></td><td>135</td></tr>
                    <tr><td>3</td><td>fuel</td><td><code>string</code></td><td>diesel</td></tr>
                    <tr><td>4</td><td>paint_color</td><td><code>string</code></td><td>black</td></tr>
                    <tr><td>5</td><td>car_type</td><td><code>string</code></td><td>sedan</td></tr>
                    <tr><td>6</td><td>private_parking_available</td><td><code>bool</code></td><td>true</td></tr>
                    <tr><td>7</td><td>has_gps</td><td><code>bool</code></td><td>true</td></tr>
                    <tr><td>8</td><td>has_air_conditioning</td><td><code>bool</code></td><td>false</td></tr>
                    <tr><td>9</td><td>automatic_car</td><td><code>bool</code></td><td>false</td></tr>
                    <tr><td>10</td><td>has_getaround_connect</td><td><code>bool</code></td><td>true</td></tr>
                    <tr><td>11</td><td>has_speed_regulator</td><td><code>bool</code></td><td>true</td></tr>
                    <tr><td>12</td><td>winter_tires</td><td><code>bool</code></td><td>true</td></tr>
                </table>
            </div>

            <div class="card">
                <h2>Exemple curl</h2>
                <pre>curl -X POST "https://athanormark-getaround-pricing-api.hf.space/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{"input": [["Renault", 140000, 135, "diesel", "black", "sedan", true, true, false, false, true, true, true]]}}'</pre>
            </div>

            <div class="card">
                <h2>Exemple Python</h2>
                <pre>import requests

response = requests.post(
    "https://athanormark-getaround-pricing-api.hf.space/predict",
    json={{"input": [["Renault", 140000, 135, "diesel", "black",
                     "sedan", True, True, False, False, True, True, True]]}}
)
print(response.json())  # {{"prediction": [139.12]}}</pre>
            </div>

            <div class="card">
                <h2>Reponse</h2>
                <p>JSON avec une cle <code>prediction</code> contenant la liste des prix predits (EUR/jour) :</p>
                <pre>{{"prediction": [139.12]}}</pre>
            </div>

            <div class="card">
                <h2>Modele</h2>
                <div class="metrics">
                    <div class="metric"><div class="value">0.75</div><div class="label">R2 Score</div></div>
                    <div class="metric"><div class="value">10.29</div><div class="label">MAE (EUR)</div></div>
                    <div class="metric"><div class="value">16.22</div><div class="label">RMSE (EUR)</div></div>
                </div>
                <p style="margin-top:12px">Gradient Boosting Regressor (scikit-learn) &mdash; 200 estimators, depth 5 &mdash; entraine sur 4 843 vehicules.</p>
            </div>
        </div>
        <div class="footer">GetAround Pricing API &mdash; Athanor Savouillan &mdash; Jedha Bootcamp</div>
    </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predit le prix journalier pour un ou plusieurs vehicules."""
    # Convertir l'input en DataFrame avec les colonnes attendues
    df = pd.DataFrame(request.input, columns=FEATURE_COLUMNS)

    # Conversion des colonnes booleennes en entiers (0/1)
    bool_cols = [
        "private_parking_available", "has_gps", "has_air_conditioning",
        "automatic_car", "has_getaround_connect", "has_speed_regulator",
        "winter_tires"
    ]
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Prediction via le pipeline sklearn
    predictions = model.predict(df)
    return {"prediction": [round(p, 2) for p in predictions]}
