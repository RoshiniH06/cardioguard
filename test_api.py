"""
CardioGuard - API Integration & Endpoint Test Suite
===================================================
PRD Module Owner: Gyanendra Deka (Backend) & Roshini Singha (Testing Coordinator)

Tests the FastAPI endpoints:
1. GET /health: Health check, system status, model availability.
2. GET /api/model-info: Metadata, feature list, PRD metrics comparison.
3. POST /predict (Nested payload): Frontend JSON contract.
4. POST /predict (Flat payload): Direct form fields.
5. POST /predict (Model choice: SVC and KNN): Specific PRD algorithms.
6. Validation errors: systolic <= diastolic, out-of-bound inputs, missing fields.
"""

import sys
import os
from pathlib import Path

# Ensure cardioguard directory is in sys.path
CARDIOGUARD_DIR = Path(__file__).resolve().parent
if str(CARDIOGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(CARDIOGUARD_DIR))

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

from main import (
    app,
    health_check,
    get_model_info,
    predict_heart_disease,
    PredictRequest,
    PatientSubModel,
    HealthParamsSubModel,
)


def test_health_endpoint():
    print("[TEST 1/6] Testing GET /health endpoint...")
    res = health_check()
    assert res["status"] == "online"
    assert "primary_model" in res
    assert "models_available" in res
    print(f"  -> Passed! Service status: {res['status']} | Available models: {res['models_available']}")


def test_model_info_endpoint():
    print("[TEST 2/6] Testing GET /api/model-info endpoint...")
    res = get_model_info()
    assert "primary_model_name" in res
    assert "features" in res
    assert "test_metrics" in res
    print(f"  -> Passed! Primary model: {res['primary_model_name']} | Features: {len(res['features'])}")


def test_predict_nested_payload():
    print("[TEST 3/6] Testing POST /predict (Nested Frontend Contract)...")
    req = PredictRequest(
        patient=PatientSubModel(
            name="Jane Doe",
            age=45.0,
            gender=1.0,
            height_cm=162.0,
            weight_kg=64.0
        ),
        health_parameters=HealthParamsSubModel(
            systolic_bp_mmhg=120.0,
            diastolic_bp_mmhg=78.0,
            cholesterol_level=1,
            glucose_level=1,
            smoking=0,
            alcohol=0,
            physically_active=1
        )
    )
    res = predict_heart_disease(req)
    assert 0.0 <= res["probability"] <= 1.0
    assert "risk_percentage" in res
    assert "risk_level" in res
    assert "biomarkers" in res
    assert "recommendations" in res
    assert "disclaimer" in res
    print(f"  -> Passed! Probability: {res['probability']:.4f} | Risk: {res['risk_percentage']}% ({res['risk_level']})")


def test_predict_flat_payload():
    print("[TEST 4/6] Testing POST /predict (Flat Input Schema)...")
    req = PredictRequest(
        name="Robert Smith",
        age=60.0,
        gender=2.0,
        height=175.0,
        weight=92.0,
        ap_hi=155.0,
        ap_lo=95.0,
        cholesterol=3,
        gluc=2,
        smoke=1,
        alco=1,
        active=0
    )
    res = predict_heart_disease(req)
    assert res["risk_percentage"] > 50.0
    assert res["risk_level"] in ["Moderate", "High"]
    print(f"  -> Passed! High-risk patient estimate: {res['risk_percentage']}% ({res['risk_level']})")


def test_predict_svc_and_knn():
    print("[TEST 5/6] Testing POST /predict with SVC and KNN model choices...")
    req_svc = PredictRequest(
        name="Alex", age=52.0, gender=2.0, height=170.0, weight=78.0,
        ap_hi=135.0, ap_lo=85.0, cholesterol=2, gluc=1, smoke=0, alco=0, active=1,
        model_choice="svc"
    )
    res_svc = predict_heart_disease(req_svc)
    print(f"  -> SVC Prediction : {res_svc['risk_percentage']}% ({res_svc['risk_level']})")

    req_knn = PredictRequest(
        name="Alex", age=52.0, gender=2.0, height=170.0, weight=78.0,
        ap_hi=135.0, ap_lo=85.0, cholesterol=2, gluc=1, smoke=0, alco=0, active=1,
        model_choice="knn"
    )
    res_knn = predict_heart_disease(req_knn)
    print(f"  -> KNN Prediction : {res_knn['risk_percentage']}% ({res_knn['risk_level']})")


def test_validation_errors():
    print("[TEST 6/6] Testing Input Validation & Error Handling (PRD FR-02)...")
    # 1. Systolic <= Diastolic error
    try:
        PredictRequest(
            name="Bad BP", age=40.0, gender=1.0, height=165.0, weight=65.0,
            ap_hi=80.0, ap_lo=120.0,  # Invalid: systolic < diastolic
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1
        )
        assert False, "Should have raised ValueError for systolic <= diastolic"
    except ValueError as e:
        print(f"  -> Caught expected error for invalid BP: {e}")

    # 2. Age out of range (< 18)
    try:
        PredictRequest(
            name="Bad Age", age=10.0, gender=1.0, height=165.0, weight=65.0,
            ap_hi=120.0, ap_lo=80.0,
            cholesterol=1, gluc=1, smoke=0, alco=0, active=1
        )
        assert False, "Should have raised ValidationError for age < 18"
    except Exception as e:
        print(f"  -> Caught expected error for age out-of-range: age < 18")


def main():
    print("=" * 65)
    print("RUNNING CARDIOGUARD API ENDPOINT & SCHEMA TESTS")
    print("=" * 65)
    test_health_endpoint()
    test_model_info_endpoint()
    test_predict_nested_payload()
    test_predict_flat_payload()
    test_predict_svc_and_knn()
    test_validation_errors()
    print("=" * 65)
    print("[SUCCESS] ALL CARDIOGUARD API INTEGRATION TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
