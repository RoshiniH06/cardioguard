"""
Quick check of the saved model (no API needed).
Run from the cardioguard folder:   python ml/test_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if not BACKEND_DIR.exists():
    BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
model = joblib.load(BACKEND_DIR / "model.joblib")
meta = json.loads((BACKEND_DIR / "meta.json").read_text())

patients = {
    "Young, healthy": dict(age_years=28, gender=1, height=165, weight=58, ap_hi=110, ap_lo=70,
                           cholesterol=1, gluc=1, smoke=0, alco=0, active=1),
    "Middle-aged, borderline": dict(age_years=50, gender=2, height=172, weight=82, ap_hi=135, ap_lo=85,
                                    cholesterol=2, gluc=1, smoke=0, alco=0, active=0),
    "Older, high risk": dict(age_years=62, gender=2, height=168, weight=95, ap_hi=160, ap_lo=100,
                             cholesterol=3, gluc=2, smoke=1, alco=0, active=0),
}

print(f"Model: {meta['model_name']} | threshold: {meta['threshold']}\n")
for name, p in patients.items():
    p["bmi"] = p["weight"] / ((p["height"] / 100) ** 2)
    X = pd.DataFrame([p])[meta["feature_columns"]]
    proba = model.predict_proba(X)[0, 1]
    label = "HEART RISK" if proba >= meta["threshold"] else "low risk"
    print(f"{name:<26} probability = {proba:.2f}  ->  {label}")
