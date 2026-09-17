from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Charger le modèle et les colonnes une seule fois, au démarrage du serveur
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))
app = FastAPI(title="Credit Risk API")
app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_methods=["*"],
       allow_headers=["*"],
   )
# Définir la structure des données qu'on attend en entrée
class ClientData(BaseModel):
    status: str
    duration: int
    credit_history: str
    purpose: str
    amount: int
    savings: str
    employment_duration: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    present_residence: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    number_credits: int
    job: str
    people_liable: int
    telephone: str
    foreign_worker: str

@app.get("/")
def home():
    return {"message": "API Credit Risk - en ligne !"}

@app.post("/predict")
def predict(client: ClientData):
    # 1. Transformer les données reçues en DataFrame (une seule ligne)
    input_df = pd.DataFrame([client.dict()])

    # 2. Appliquer le même One-Hot Encoding qu'à l'entraînement
    input_encoded = pd.get_dummies(input_df)

    # 3. Ajouter les colonnes manquantes (celles vues à l'entraînement mais absentes ici)
    for col in columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # 4. Garder uniquement les colonnes attendues, dans le bon ordre
    input_encoded = input_encoded[columns]

    # 5. Prédire
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]

    result = "Bon payeur" if prediction == 1 else "Mauvais payeur"

    return {
        "prediction": int(prediction),
        "resultat": result,
        "probabilite_bon_payeur": round(float(probability[1]), 3),
        "probabilite_mauvais_payeur": round(float(probability[0]), 3)
    }