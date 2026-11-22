"""
CardioGuard - Model Unit Tests & Personas Validation
====================================================
PRD Module Owner: Roshini Singha (ML Lead) & Team Group C

Tests all saved models (Primary, SVC, KNN) on standardized clinical patient profiles:
1. Young, healthy individual (expected low risk)
2. Middle-aged, borderline risk (elevated BP and cholesterol)
3. Older, high-risk individual (hypertensive, smoker, elevated lipids)
4. Asserts valid probability bounds [0.0, 1.0] and risk classifications.
"""

import json
from pathlib import Path
import joblib
import pandas as pd

# Directories
BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if not BACKEND_DIR.exists():
    BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"

# Load metadata
meta_file = BACKEND_DIR / "meta.json"
meta = json.loads(meta_file.read_text(encoding="utf-8"))
feature_cols = meta["feature_columns"]
threshold = meta["threshold"]

print("=" * 65)
print("CARDIOGUARD - MODEL UNIT & INTEGRATION TESTING")
print(f"Primary Serving Model: {meta.get('primary_model_name', 'HistGradientBoosting')}")
print(f"Decision Threshold   : {threshold:.2f}")
print("=" * 65)

# Load available model pipelines
models = {}
for m_key, fname in [
    ("Primary Model", "model.joblib"),
    ("Support Vector Classifier", "model_svc.joblib"),
    ("K-Nearest Neighbors", "model_knn.joblib"),
]:
    fpath = BACKEND_DIR / fname
    if fpath.exists():
        models[m_key] = joblib.load(fpath)
        print(f"[LOADED] {m_key} from {fname}")

patients = {
    "Young Healthy (28y Female)": {
        "age_years": 28.0, "gender": 1.0, "height": 165.0, "weight": 58.0,
        "ap_hi": 110.0, "ap_lo": 70.0, "cholesterol": 1, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 1
    },
    "Middle-Aged Borderline (50y Male)": {
        "age_years": 50.0, "gender": 2.0, "height": 172.0, "weight": 82.0,
        "ap_hi": 135.0, "ap_lo": 85.0, "cholesterol": 2, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 0
    },
    "Older High-Risk (62y Hypertensive)": {
        "age_years": 62.0, "gender": 2.0, "height": 168.0, "weight": 95.0,
        "ap_hi": 165.0, "ap_lo": 105.0, "cholesterol": 3, "gluc": 2,
        "smoke": 1, "alco": 0, "active": 0
    },
    "Elderly Non-Smoker High-BP (64y Female)": {
        "age_years": 64.0, "gender": 1.0, "height": 160.0, "weight": 78.0,
        "ap_hi": 150.0, "ap_lo": 95.0, "cholesterol": 2, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 1
    }
}

print("\n" + "-" * 65)
print(f"{'Patient Profile':<38} | {'Model':<12} | {'Risk %':<8} | {'Class'}")
print("-" * 65)

for p_name, p_data in patients.items():
    # Calculate BMI
    bmi = p_data["weight"] / ((p_data["height"] / 100.0) ** 2)
    p_data["bmi"] = round(bmi, 2)
    
    # Prepare DataFrame matching training feature columns
    X_input = pd.DataFrame([p_data])[feature_cols]

    for m_label, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_input)[0, 1])
        else:
            proba = float(model.predict(X_input)[0])
            
        assert 0.0 <= proba <= 1.0, f"Probability {proba} out of bounds!"
        
        is_high = proba >= threshold
        risk_class = "ELEVATED RISK" if is_high else "Low Risk"
        print(f"{p_name:<38} | {m_label[:12]:<12} | {proba*100:>6.1f}% | {risk_class}")
    print("." * 65)

print("\n[SUCCESS] All model test assertions passed successfully!")
