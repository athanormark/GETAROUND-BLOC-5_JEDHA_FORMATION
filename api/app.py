import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="GetAround Pricing API",
    description="API de prediction du prix journalier de location de voiture",
    version="1.0.0"
)

# Chargement du modele
model = joblib.load("model.joblib")

# Colonnes attendues
FEATURE_COLUMNS = [
    "model_key", "mileage", "engine_power", "fuel", "paint_color",
    "car_type", "private_parking_available", "has_gps", "has_air_conditioning",
    "automatic_car", "has_getaround_connect", "has_speed_regulator", "winter_tires"
]

class PredictionInput(BaseModel):
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
    input: List[List]

class PredictionResponse(BaseModel):
    prediction: List[float]


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><title>GetAround Pricing API</title></head>
    <body>
    <h1>GetAround Pricing API</h1>
    <p>API de prediction du prix journalier de location.</p>
    <p><a href="/docs">Documentation</a></p>
    </body></html>
    """


@app.get("/docs", response_class=HTMLResponse)
def documentation():
    return """
    <html>
    <head>
        <title>GetAround API - Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
            code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
            pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>GetAround Pricing API - Documentation</h1>

        <h2>POST /predict</h2>
        <p>Predit le prix journalier de location a partir des caracteristiques du vehicule.</p>

        <h3>Methode HTTP</h3>
        <p><code>POST</code></p>

        <h3>Input</h3>
        <p>JSON avec une cle <code>input</code> contenant une liste de listes. Chaque sous-liste represente un vehicule avec 13 features dans l'ordre :</p>

        <table>
            <tr><th>#</th><th>Feature</th><th>Type</th><th>Exemple</th></tr>
            <tr><td>0</td><td>model_key</td><td>string</td><td>"Renault"</td></tr>
            <tr><td>1</td><td>mileage</td><td>int</td><td>140000</td></tr>
            <tr><td>2</td><td>engine_power</td><td>int</td><td>135</td></tr>
            <tr><td>3</td><td>fuel</td><td>string</td><td>"diesel"</td></tr>
            <tr><td>4</td><td>paint_color</td><td>string</td><td>"black"</td></tr>
            <tr><td>5</td><td>car_type</td><td>string</td><td>"sedan"</td></tr>
            <tr><td>6</td><td>private_parking_available</td><td>bool</td><td>true</td></tr>
            <tr><td>7</td><td>has_gps</td><td>bool</td><td>true</td></tr>
            <tr><td>8</td><td>has_air_conditioning</td><td>bool</td><td>false</td></tr>
            <tr><td>9</td><td>automatic_car</td><td>bool</td><td>false</td></tr>
            <tr><td>10</td><td>has_getaround_connect</td><td>bool</td><td>true</td></tr>
            <tr><td>11</td><td>has_speed_regulator</td><td>bool</td><td>true</td></tr>
            <tr><td>12</td><td>winter_tires</td><td>bool</td><td>true</td></tr>
        </table>

        <h3>Exemple de requete</h3>
        <pre>
curl -X POST "https://your-url/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"input": [["Renault", 140000, 135, "diesel", "black", "sedan", true, true, false, false, true, true, true]]}'
        </pre>

        <h3>Exemple avec Python</h3>
        <pre>
import requests

response = requests.post("https://your-url/predict", json={
    "input": [["Renault", 140000, 135, "diesel", "black", "sedan", True, True, False, False, True, True, True]]
})
print(response.json())
        </pre>

        <h3>Output</h3>
        <p>JSON avec une cle <code>prediction</code> contenant la liste des prix predits (en EUR/jour).</p>
        <pre>{"prediction": [138.0]}</pre>

        <h2>GET /</h2>
        <p>Page d'accueil de l'API.</p>

        <h2>Modele</h2>
        <p>Gradient Boosting Regressor (scikit-learn) entraine sur 4843 vehicules.</p>
        <p>Metriques sur le jeu de test : R2 = 0.75, MAE = 10.29 EUR</p>
    </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Convertir l'input en DataFrame
    df = pd.DataFrame(request.input, columns=FEATURE_COLUMNS)

    # Conversion des types booleens
    bool_cols = [
        "private_parking_available", "has_gps", "has_air_conditioning",
        "automatic_car", "has_getaround_connect", "has_speed_regulator", "winter_tires"
    ]
    for col in bool_cols:
        df[col] = df[col].astype(int)

    predictions = model.predict(df)
    return {"prediction": [round(p, 2) for p in predictions]}
