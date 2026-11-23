"""
CardioGuard - FastAPI Backend API
==================================
PRD Module Owner: Gyanendra Deka (Backend Development)
Team: Group C | BCA 1st Semester | Scholars Institute of Technology and Management

This backend:
1. Exposes /predict endpoint for heart disease risk estimation.
2. Accepts both nested schema (from frontend contract) and flat schema.
3. Performs rigorous input validation per PRD FR-02.
4. Loads trained machine-learning pipelines (Calibrated SVC, KNN, and Ensemble).
5. Returns probability, risk category, personalized biomarker assessment, and mandatory medical disclaimer.
6. Exposes /api/model-info for viva transparency and model comparison inspection.
7. Serves the frontend static files (HTML, CSS, JS) at the root URL.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, root_validator, model_validator

# -------------------------------------------------------------------------
# DIRECTORY & APP INITIALIZATION
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"

# Locate frontend files (supports dedicated frontend/ directory and fallback to base)
FRONTEND_DIR = (BASE_DIR / "frontend") if (BASE_DIR / "frontend").exists() else BASE_DIR
CSS_DIR = (FRONTEND_DIR / "css") if (FRONTEND_DIR / "css").exists() else (BASE_DIR / "css")
JS_DIR = (FRONTEND_DIR / "js") if (FRONTEND_DIR / "js").exists() else (BASE_DIR / "js")

app = FastAPI(
    title="CardioGuard API",
    description="Early Cardiovascular Disease Risk Prediction Portal Backend (PRD Group C)",
    version="1.0.0",
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# LOAD TRAINED ML MODELS & METADATA
# -------------------------------------------------------------------------
models_dict: Dict[str, Any] = {}
meta: Dict[str, Any] = {}

try:
    # 1. Primary Model
    primary_model_path = BACKEND_DIR / "model.joblib"
    if primary_model_path.exists():
        models_dict["primary"] = joblib.load(primary_model_path)
    
    # 2. Support Vector Classifier (PRD Requirement)
    svc_model_path = BACKEND_DIR / "model_svc.joblib"
    if svc_model_path.exists():
        models_dict["svc"] = joblib.load(svc_model_path)
        
    # 3. K-Nearest Neighbors (PRD Requirement)
    knn_model_path = BACKEND_DIR / "model_knn.joblib"
    if knn_model_path.exists():
        models_dict["knn"] = joblib.load(knn_model_path)

    # 4. Metadata
    meta_path = BACKEND_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {
            "primary_model_name": "Support Vector Classifier",
            "threshold": 0.40,
            "high_risk_probability": 0.65,
            "feature_columns": [
                "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
                "cholesterol", "gluc", "smoke", "alco", "active", "bmi"
            ]
        }
        
    PRIMARY_MODEL_NAME = meta.get("primary_model_name", "CardioGuard Model")
    THRESHOLD = meta.get("threshold", 0.40)
    HIGH_RISK_THRESHOLD = meta.get("high_risk_probability", 0.65)
    FEATURE_COLS = meta.get("feature_columns", [
        "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", "bmi"
    ])
    print(f"[OK] Loaded CardioGuard models: {list(models_dict.keys())} | Threshold: {THRESHOLD}")

except Exception as err:
    print(f"[WARNING] Model loading notice: {err}")
    models_dict = {}


# -------------------------------------------------------------------------
# REQUEST SCHEMAS (Supporting Nested & Flat Payloads)
# -------------------------------------------------------------------------
class PatientSubModel(BaseModel):
    name: Optional[str] = "Patient"
    age: float = Field(..., ge=18, le=120, description="Age in years (30-65 in training set)")
    gender: float = Field(..., description="1 = Female, 2 = Male, 1.5 = Other")
    height_cm: float = Field(..., ge=80, le=250, description="Height in cm")
    weight_kg: float = Field(..., ge=20, le=300, description="Weight in kg")


class HealthParamsSubModel(BaseModel):
    systolic_bp_mmhg: float = Field(..., ge=50, le=300, description="Systolic blood pressure (ap_hi)")
    diastolic_bp_mmhg: float = Field(..., ge=30, le=200, description="Diastolic blood pressure (ap_lo)")
    cholesterol_level: int = Field(..., ge=1, le=3, description="1: Normal, 2: Above Normal, 3: Well Above Normal")
    glucose_level: int = Field(..., ge=1, le=3, description="1: Normal, 2: Above Normal, 3: Well Above Normal")
    smoking: int = Field(..., ge=0, le=1, description="0: Non-smoker, 1: Smoker")
    alcohol: int = Field(..., ge=0, le=1, description="0: No alcohol, 1: Alcohol consumer")
    physically_active: int = Field(..., ge=0, le=1, description="0: Not active, 1: Physically active")


class PredictRequest(BaseModel):
    # Optional nested structure
    patient: Optional[PatientSubModel] = None
    health_parameters: Optional[HealthParamsSubModel] = None

    # Optional flat structure (for simple API testing or direct form posts)
    name: Optional[str] = None
    age: Optional[float] = None
    gender: Optional[float] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    ap_hi: Optional[float] = None
    ap_lo: Optional[float] = None
    cholesterol: Optional[int] = None
    gluc: Optional[int] = None
    smoke: Optional[int] = None
    alco: Optional[int] = None
    active: Optional[int] = None

    # Model selector: "primary", "svc", or "knn"
    model_choice: Optional[str] = "primary"

    @model_validator(mode="after")
    def validate_and_unify_fields(self):
        # Extract from nested if available
        if self.patient is not None and self.health_parameters is not None:
            p = self.patient
            h = self.health_parameters
            self.name = self.name or p.name or "Patient"
            self.age = p.age
            self.gender = p.gender
            self.height = p.height_cm
            self.weight = p.weight_kg
            self.ap_hi = h.systolic_bp_mmhg
            self.ap_lo = h.diastolic_bp_mmhg
            self.cholesterol = h.cholesterol_level
            self.gluc = h.glucose_level
            self.smoke = h.smoking
            self.alco = h.alcohol
            self.active = h.physically_active

        # Validate that essential attributes are now present
        required_fields = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
        missing = [f for f in required_fields if getattr(self, f, None) is None]
        if missing:
            raise ValueError(f"Missing required patient parameters: {missing}")

        # Validate blood pressure logic
        if self.ap_hi <= self.ap_lo:
            raise ValueError(f"Systolic blood pressure ({self.ap_hi} mmHg) must be greater than diastolic blood pressure ({self.ap_lo} mmHg).")

        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convert validated inputs into the exact feature format required by the model."""
        height_m = self.height / 100.0
        bmi = round(self.weight / (height_m ** 2), 2)
        
        # Gender normalization (if 1.5, map to 1 or 2 average)
        gender_val = 1 if self.gender == 1 else (2 if self.gender == 2 else 1.5)

        data = {
            "age_years": [float(self.age)],
            "gender": [float(gender_val)],
            "height": [float(self.height)],
            "weight": [float(self.weight)],
            "ap_hi": [float(self.ap_hi)],
            "ap_lo": [float(self.ap_lo)],
            "cholesterol": [int(self.cholesterol)],
            "gluc": [int(self.gluc)],
            "smoke": [int(self.smoke)],
            "alco": [int(self.alco)],
            "active": [int(self.active)],
            "bmi": [float(bmi)],
        }
        return pd.DataFrame(data)[FEATURE_COLS]


# -------------------------------------------------------------------------
# HELPER LOGIC FOR CLINICAL BIOMARKERS & RECOMMENDATIONS
# -------------------------------------------------------------------------
def categorize_bp(ap_hi: float, ap_lo: float) -> Dict[str, str]:
    if ap_hi > 180 or ap_lo > 120:
        return {"category": "Hypertensive Crisis", "status": "Critical", "note": "Emergency medical consultation advised if accompanied by symptoms."}
    elif ap_hi >= 140 or ap_lo >= 90:
        return {"category": "Hypertension Stage 2", "status": "High", "note": "Consistent readings in this range indicate stage 2 hypertension."}
    elif ap_hi >= 130 or ap_lo >= 80:
        return {"category": "Hypertension Stage 1", "status": "Elevated", "note": "Discuss repeated readings with a healthcare provider."}
    elif ap_hi >= 120 and ap_lo < 80:
        return {"category": "Elevated Blood Pressure", "status": "Borderline", "note": "Heart-healthy diet and active lifestyle recommended."}
    else:
        return {"category": "Normal Blood Pressure", "status": "Optimal", "note": "Blood pressure is in the standard healthy adult range."}


def categorize_bmi(bmi: float) -> Dict[str, str]:
    if bmi < 18.5:
        return {"category": "Underweight", "status": "Caution"}
    elif bmi < 25.0:
        return {"category": "Normal weight", "status": "Optimal"}
    elif bmi < 30.0:
        return {"category": "Overweight", "status": "Elevated"}
    else:
        return {"category": "Obesity", "status": "High"}


def generate_personalized_recommendations(patient_req: PredictRequest, proba: float, bmi: float) -> list:
    tips = []
    
    # Blood pressure tips
    if patient_req.ap_hi >= 130 or patient_req.ap_lo >= 80:
        tips.append("Regularly monitor blood pressure, reduce dietary sodium intake (<2,300 mg/day), and consult a doctor.")
    else:
        tips.append("Maintain routine blood pressure checks annually to ensure readings remain in the healthy range.")

    # BMI tips
    if bmi >= 25.0:
        tips.append(f"Current BMI is {bmi:.1f} kg/m² (overweight/obese range). Aim for gradual, sustainable caloric balance and fiber-rich meals.")
    elif bmi < 18.5:
        tips.append("Current BMI is below the healthy range. Discuss adequate nutrient intake with a qualified nutritionist.")

    # Cholesterol & Glucose tips
    if patient_req.cholesterol > 1:
        tips.append("Self-reported cholesterol is above normal. Schedule a lipid panel blood test with your healthcare provider.")
    if patient_req.gluc > 1:
        tips.append("Self-reported glucose is elevated. Consider an HbA1c or fasting blood glucose check.")

    # Lifestyle tips
    if patient_req.smoke == 1:
        tips.append("Smoking significantly increases cardiovascular risk. Seek professional cessation counseling or nicotine-replacement therapies.")
    if patient_req.alco == 1:
        tips.append("Limit alcohol consumption as excessive intake elevates blood pressure and cardiac strain.")
    if patient_req.active == 0:
        tips.append("Incorporate at least 150 minutes of moderate aerobic activity (e.g. brisk walking) weekly if safe for you.")
    else:
        tips.append("Great job staying physically active! Continue maintaining consistent cardiovascular exercise.")

    return tips


# -------------------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------------------
@app.get("/health")
@app.get("/api/status")
def health_check():
    """System health and model status endpoint."""
    return {
        "status": "online",
        "service": "CardioGuard Cardiovascular Disease Risk Prediction Portal",
        "models_available": list(models_dict.keys()),
        "primary_model": PRIMARY_MODEL_NAME,
        "default_threshold": THRESHOLD,
        "team": {
            "group": "Group C",
            "institution": "Scholars Institute of Technology and Management",
            "programme": "BCA - 1st Semester",
            "leader": "Roshini Singha",
            "backend_lead": "Gyanendra Deka",
            "data_prep": "Madhurjya Dewraja",
            "frontend_lead": "Manash Sarma"
        }
    }


@app.get("/api/model-info")
def get_model_info():
    """Return model evaluation metrics, PRD comparison (SVC vs KNN), and dataset details."""
    return {
        "primary_model_name": PRIMARY_MODEL_NAME,
        "threshold": THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "features": FEATURE_COLS,
        "dataset_metadata": meta.get("dataset", {
            "name": "Cardiovascular Disease Dataset (cardio_train.csv)",
            "records": 68586,
            "target": "Presence (1) vs Absence (0) of Cardiovascular Disease"
        }),
        "test_metrics": meta.get("test_metrics", {}),
        "model_comparison": meta.get("all_model_comparison", []),
        "disclaimer": "Academic demonstration only. Not for clinical diagnosis."
    }


@app.post("/predict")
def predict_heart_disease(request: PredictRequest):
    """
    Core prediction endpoint.
    Accepts validated patient health attributes and computes:
    - Probability of CVD (0.0 to 1.0)
    - Risk Level (Low, Moderate, High)
    - Classification (0 = Low Risk, 1 = Elevated Risk)
    - Biomarker analysis and personalized suggestions
    """
    try:
        # Convert request to model DataFrame
        df_input = request.to_dataframe()
        height_m = request.height / 100.0
        bmi = round(request.weight / (height_m ** 2), 2)

        # Select model (Primary, SVC, or KNN)
        choice = request.model_choice.lower() if request.model_choice else "primary"
        model_to_use = models_dict.get(choice) or models_dict.get("primary")

        if model_to_use is None:
            # Fallback mathematical logistic model if no model loaded
            # Logistic formula trained on cardio_train:
            # intercept: 0.0277
            # weights for age, gender, height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active
            means = [53.28, 1.35, 164.4, 74.14, 126.6, 81.28, 1.37, 1.23, 0.088, 0.054, 0.803]
            scales = [6.77, 0.48, 7.90, 14.34, 16.70, 9.44, 0.68, 0.57, 0.28, 0.23, 0.40]
            coefs = [0.348, -0.009, -0.025, 0.153, 0.930, 0.112, 0.341, -0.066, -0.043, -0.052, -0.090]
            intercept = 0.0277
            
            raw_vals = [request.age, request.gender, request.height, request.weight, request.ap_hi, request.ap_lo,
                        request.cholesterol, request.gluc, request.smoke, request.alco, request.active]
            z = intercept
            for v, m, s, c in zip(raw_vals, means, scales, coefs):
                z += ((v - m) / s) * c
            proba = float(1.0 / (1.0 + np.exp(-z)))
            model_used_name = "Logistic Regression (Built-in Fallback)"
        else:
            proba_arr = model_to_use.predict_proba(df_input)
            proba = float(proba_arr[0][1])
            model_used_name = choice.upper() if choice in ["svc", "knn"] else PRIMARY_MODEL_NAME

        # Calculate risk categories
        risk_pct = round(proba * 100, 1)
        is_disease_pred = int(proba >= THRESHOLD)

        if proba >= HIGH_RISK_THRESHOLD:
            risk_level = "High"
            risk_badge = "High Risk"
            summary_msg = f"Higher model estimate ({risk_pct}%). Elevated cardiovascular indicators identified."
        elif proba >= THRESHOLD:
            risk_level = "Moderate"
            risk_badge = "Moderate Risk"
            summary_msg = f"Moderate model estimate ({risk_pct}%). Some indicators are above baseline ranges."
        else:
            risk_level = "Low"
            risk_badge = "Low Risk"
            summary_msg = f"Lower model estimate ({risk_pct}%). Entered health attributes are largely within expected ranges."

        # Biomarker evaluations
        bp_eval = categorize_bp(request.ap_hi, request.ap_lo)
        bmi_eval = categorize_bmi(bmi)
        recommendations = generate_personalized_recommendations(request, proba, bmi)

        return {
            "patient_name": request.name or "Patient",
            "probability": round(proba, 4),
            "risk_percentage": risk_pct,
            "risk_level": risk_level,
            "risk_badge": risk_badge,
            "heart_disease_prediction": is_disease_pred,
            "threshold_used": THRESHOLD,
            "model_name": model_used_name,
            "biomarkers": {
                "age_years": request.age,
                "blood_pressure": f"{int(request.ap_hi)}/{int(request.ap_lo)} mmHg",
                "blood_pressure_category": bp_eval["category"],
                "bmi": bmi,
                "bmi_category": bmi_eval["category"],
                "cholesterol_category": ["", "Normal", "Above Normal", "Well Above Normal"][request.cholesterol],
                "glucose_category": ["", "Normal", "Above Normal", "Well Above Normal"][request.gluc],
                "smoker": bool(request.smoke),
                "alcohol": bool(request.alco),
                "physically_active": bool(request.active)
            },
            "recommendations": recommendations,
            "message": summary_msg,
            "disclaimer": (
                "Educational Screening Notice: This prediction is generated by an academic machine learning prototype "
                "(Group C BCA End-Semester Project) trained on the public Cardiovascular Disease dataset. It is not a clinical "
                "diagnosis or medical advice. Individuals should consult a qualified healthcare provider for proper evaluation."
            )
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# -------------------------------------------------------------------------
# SERVE FRONTEND STATIC FILES
# -------------------------------------------------------------------------
if CSS_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(CSS_DIR)), name="css")

if JS_DIR.exists():
    app.mount("/js", StaticFiles(directory=str(JS_DIR)), name="js")


@app.get("/")
@app.get("/index.html")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(str(index_file))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
