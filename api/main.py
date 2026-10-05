from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent


def find_file(name: str) -> Path:
    for p in (BASE_DIR / name, BASE_DIR / "models" / name, BASE_DIR.parent / "models" / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} introuvable")


model = joblib.load(find_file("model.pkl"))
columns = joblib.load(find_file("columns.pkl"))

app = FastAPI(title="Credit Risk API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"message": "API Credit Risk - en ligne !"}


@app.post("/predict")
def predict(client: ClientData):
    input_df = pd.DataFrame([client.dict()])
    input_encoded = pd.get_dummies(input_df)

    for col in columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[columns]

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]
    result = "Bon payeur" if prediction == 1 else "Mauvais payeur"

    return {
        "prediction": int(prediction),
        "resultat": result,
        "probabilite_bon_payeur": round(float(probability[1]), 3),
        "probabilite_mauvais_payeur": round(float(probability[0]), 3),
    }