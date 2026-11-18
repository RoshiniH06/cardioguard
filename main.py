from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from pathlib import Path

app = FastAPI(title="CardioGuard API", description="Cardiovascular Disease Prediction API")

# --- THIS ALLOWS YOUR FRONTEND TO CONNECT ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Load the Model and Metadata on Startup ---
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"

try:
    model = joblib.load(BACKEND_DIR / "model.joblib")
    with open(BACKEND_DIR / "meta.json", "r") as f:
        meta = json.load(f)
    THRESHOLD = meta["threshold"]
    HIGH_RISK_PROBA = meta["high_risk_probability"]
except Exception as e:
    raise RuntimeError(f"Failed to load model files. Error: {e}")

# --- 2. Define the Input Data Schema ---
class PatientData(BaseModel):
    General_Health: str
    Checkup: str
    Exercise: str
    Diabetes: str
    Depression: str
    Age_Category: str
    Sex: str
    Smoking_History: str
    Skin_Cancer: str
    Other_Cancer: str
    Arthritis: str
    BMI: float
    Alcohol_Consumption: float
    Fruit_Consumption: float
    Green_Vegetables_Consumption: float
    FriedPotato_Consumption: float
    Height_cm: float = 170.0
    Weight_kg: float = 70.0

    def to_dataframe(self):
        data = self.model_dump()
        data["Height_(cm)"] = data.pop("Height_cm")
        data["Weight_(kg)"] = data.pop("Weight_kg")
        return pd.DataFrame([data])

# --- 3. The Prediction Endpoint ---
@app.post("/predict")
def predict_heart_disease(patient: PatientData):
    try:
        df = patient.to_dataframe()
        proba = float(model.predict_proba(df)[0][1])
        is_high_risk = proba >= THRESHOLD
        
        if proba >= HIGH_RISK_PROBA:
            risk_level = "High"
        elif proba >= THRESHOLD:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return {
            "probability": round(proba, 4),
            "threshold_used": THRESHOLD,
            "heart_disease_prediction": int(is_high_risk),
            "risk_level": risk_level,
            "message": "High risk of heart disease. Please consult a doctor." if is_high_risk else "Low risk detected."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- 4. Serve the Frontend Files ---
# Mount CSS and JS directories robustly using BASE_DIR
app.mount("/css", StaticFiles(directory=str(BASE_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(BASE_DIR / "js")), name="js")

# When you go to http://127.0.0.1:8000, load index.html
@app.get("/")
def serve_frontend():
    return FileResponse(str(BASE_DIR / "index.html"))