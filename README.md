# CardioGuard: Early Cardiovascular Disease Risk Prediction Portal

**Scholars Institute of Technology and Management**  
**Department of Computer Science & Applications**  
**Course & Semester:** BCA – 1st Semester | November 2026  
**Assessment:** Project-Based End-Semester Assessment (PRD Group C)  
**Faculty Mentor:** Mr. Soumyajit Jalua  

---

## 👥 Team Member Register & Module Ownership

| No. | Student Name | Role & Ownership | Contact |
|:---:|:---|:---|:---:|
| 1 | **Roshini Singha** | **Team Leader & ML Lead** (Pipelines, SVC & KNN Training, Model Selection & Evaluation) | 6900719986 |
| 2 | **Gyanendra Deka** | **Backend Lead** (FastAPI Architecture, Input Validation, Inference Endpoints) | 8822644496 |
| 3 | **Madhurjya Dewraja** | **Data Preparation Lead** (Data Quality Checks, Cleaning Outliers, Feature Engineering & EDA) | 7099413568 |
| 4 | **Manash Sarma** | **Frontend Lead** (Responsive Portal UI, Form Validation, Live Dashboard & Biomarkers) | 9181861626 |

---

## 📋 Project Summary

**CardioGuard** is an end-to-end academic prototype for early cardiovascular disease (CVD) risk awareness and preventive screening. The system allows users to complete a clinical health screening form entering demographic and biomarker data (Age, Gender, Height, Weight, Systolic BP, Diastolic BP, Cholesterol, Glucose, Smoking status, Alcohol consumption, and Physical activity). 

The frontend sends validated inputs to a high-performance **Python FastAPI** backend which queries trained, leakage-free machine-learning pipelines (**Support Vector Classifier (SVC)**, **K-Nearest Neighbors (KNN)**, and a **Tree Ensemble** benchmark) trained on the 70,000-record Cardiovascular Disease dataset. The portal returns calibrated risk estimates, biomarker evaluations (AHA BP stage, CDC BMI category), personalized preventive recommendations, downloadable PDF health summaries, and an explicit academic non-diagnostic disclaimer.

---

## 🏛 System Architecture

```text
       USER / CLINICAL DEMONSTRATOR
                   │ Enters health attributes & records
                   ▼
  BROWSER FRONTEND (Manash Sarma)
  ├── HTML5 / CSS3 / JavaScript
  ├── Input validation (FR-01, FR-02)
  ├── Camera / Folder OCR Patient Card Scanning (Tesseract.js)
  ├── Interactive AHA Blood Pressure & CDC BMI Calculators
  └── Dynamic SVG Dashboard & Printable / Downloadable PDF Report (jsPDF)
                   │
                   │ HTTP POST (JSON Payload: nested or flat)
                   ▼
  FASTAPI BACKEND (Gyanendra Deka)
  ├── CORS & Pydantic Request Validation (FR-03)
  ├── Out-of-bounds & physiological sanity checks (Systolic > Diastolic)
  ├── Route: POST /predict (Inference)
  ├── Route: GET /health (Service status)
  ├── Route: GET /api/model-info (Model comparison & PRD metrics)
  └── Static files serving (HTML, CSS, JS)
                   │
                   │ Evaluates feature vector
                   ▼
  TRAINED ML PIPELINES (Roshini Singha & Madhurjya Dewraja)
  ├── Data Preprocessing & Cleaning (68,586 clean records)
  ├── StandardScaler fitted strictly on training folds (Leakage-free)
  ├── Models: Support Vector Classifier (SVC) & K-Nearest Neighbors (KNN)
  ├── Benchmark: HistGradientBoosting & Logistic Regression
  └── Threshold optimization for clinical recall & sensitivity
                   │
                   ▼
  PREDICTION RESPONSE (JSON)
  ├── Probability (0.0 to 1.0) & Risk Percentage (0-100%)
  ├── Risk Level: Low, Moderate, or High
  ├── Active Algorithm Used (Ensemble, SVC, or KNN)
  ├── Biomarker Category Assessments & Personalized Advice
  └── Non-Diagnostic Academic Disclaimer (FR-05)
```

---

## 📊 Dataset & Exploratory Data Analysis (EDA)

- **Dataset Source:** Cardiovascular Diseases Risk Dataset (`cardio_train.csv`)
- **Total Raw Records:** 70,000 patients
- **Data Cleaning (Madhurjya Dewraja):**
  - Converted age in days to `age_years = age / 365.25`.
  - Calculated Body Mass Index: `bmi = weight / ((height / 100) ** 2)`.
  - Filtered physiological anomalies (Systolic BP 60–240 mmHg, Diastolic BP 40–150 mmHg, Systolic > Diastolic, Height 120–220 cm, Weight 30–200 kg, BMI 12–60).
  - Retained **68,586 valid records** (97.98% retention rate).
- **Target Balance:**
  - Class 0 (No CVD): 34,659 (50.53%)
  - Class 1 (CVD Present): 33,927 (49.47%)
- **EDA Artifacts Generated:**
  - `ml/outputs/eda_target_distribution.png`
  - `ml/outputs/eda_blood_pressure.png`
  - `ml/outputs/eda_features_correlation.png`
  - `ml/outputs/eda_summary.json`

---

## 🤖 Machine Learning Models & Comparative Evaluation

Models were evaluated on a held-out 20% stratified test split (13,718 records) per PRD Section 3 & 8:

| Model Name | Accuracy | Recall (Sensitivity) | Precision | Specificity | F1-Score | ROC-AUC | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Dummy Baseline** | 50.53% | 0.00% | 0.00% | 100.0% | 0.00% | 0.5000 | Reference baseline |
| **Logistic Regression** | 72.64% | 66.52% | 75.31% | 78.63% | 70.64% | 0.7906 | Linear benchmark |
| **K-Nearest Neighbors (KNN k=15)** | 71.71% | 68.27% | 72.84% | 75.07% | 70.48% | 0.7764 | **PRD Candidate O3** |
| **Support Vector Classifier (Calibrated SVC)** | **72.67%** | **66.80%** | **75.20%** | **78.42%** | **70.75%** | **0.7905** | **PRD Candidate O3** |
| **HistGradientBoosting (Default)** | 73.39% | 68.60% | 75.38% | 78.07% | 71.83% | **0.8009** | Tree ensemble |
| **HistGradientBoosting (Tuned thr=0.39)** | 72.24% | **78.84%** | 69.29% | 65.79% | **73.75%** | **0.8009** | **Primary Serving Model** |

### Evaluation Plots
- **ROC Curves:** Saved in `ml/outputs/roc_curve.png`
- **Confusion Matrices:** Saved in `ml/outputs/confusion_matrices.png`
- **Model Comparison Table:** Saved in `ml/outputs/model_comparison.csv`

---

## 📁 Repository Directory Structure

```text
cardioguard/
├── frontend/                    <- Dedicated standalone frontend (Deployable to Netlify / GitHub Pages)
│   ├── index.html               <- Main application portal (Intake, Dashboard, Calculators, Reports)
│   ├── css/
│   │   └── styles.css           <- Full responsive CSS stylesheet with embedded visual assets
│   ├── js/
│   │   ├── config.js            <- Configurable backend API URL setting
│   │   └── script.js            <- Dynamic JS controller (API client, PDF generator, OCR scanner)
│   └── README.md                <- Frontend standalone deployment guide
├── backend/
│   ├── meta.json                <- Model thresholds, features, and comparative metrics
│   ├── model.joblib             <- Primary production serving pipeline
│   ├── model_svc.joblib         <- Calibrated Support Vector Classifier pipeline (PRD O3)
│   └── model_knn.joblib         <- K-Nearest Neighbors pipeline (PRD O3)
├── ml/
│   ├── cardio_train.csv         <- Raw dataset
│   ├── cardio_cleaned.csv       <- Prepared, validated dataset
│   ├── data_prep.py             <- Data cleaning and EDA module (Madhurjya)
│   └── outputs/                 <- Evaluation charts, ROC plots, and confusion matrices
├── main.py                      <- FastAPI application entrypoint (Gyanendra)
├── train_model.py               <- Leakage-free ML training & comparison script (Roshini)
├── test_model.py                <- Model unit test assertions across clinical personas
├── test_api.py                  <- FastAPI integration and validation test suite
└── requirements.txt             <- Python project dependencies
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```powershell
cd cardioguard
pip install -r requirements.txt
```

### 2. Run Data Cleaning & Exploratory Analysis
```powershell
python ml/data_prep.py
```

### 3. Train Candidate Models & Generate Metrics
```powershell
python train_model.py
```

### 4. Run Model Unit Tests
```powershell
python test_model.py
```

### 5. Run API Integration Tests
```powershell
python test_api.py
```

### 6. Start the Web Portal & FastAPI Server
```powershell
python main.py
```
*Or using uvicorn directly:*
```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then open your browser and navigate to:
- **Interactive Web Portal:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Model Info Endpoint:** [http://127.0.0.1:8000/api/model-info](http://127.0.0.1:8000/api/model-info)
- **API Health Status:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🔌 API Reference

### `POST /predict`
Evaluates cardiovascular disease risk probability.

#### Example Request (JSON):
```json
{
  "patient": {
    "name": "Jane Doe",
    "age": 52,
    "gender": 1,
    "height_cm": 164,
    "weight_kg": 72
  },
  "health_parameters": {
    "systolic_bp_mmhg": 138,
    "diastolic_bp_mmhg": 88,
    "cholesterol_level": 2,
    "glucose_level": 1,
    "smoking": 0,
    "alcohol": 0,
    "physically_active": 1
  },
  "model_choice": "primary"
}
```
*(Flat payload format with direct fields `age`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`, etc. is also fully supported).*

#### Example Response (JSON):
```json
{
  "patient_name": "Jane Doe",
  "probability": 0.5842,
  "risk_percentage": 58.4,
  "risk_level": "Moderate",
  "risk_badge": "Moderate Risk",
  "heart_disease_prediction": 1,
  "threshold_used": 0.39,
  "model_name": "HistGradientBoosting",
  "biomarkers": {
    "age_years": 52.0,
    "blood_pressure": "138/88 mmHg",
    "blood_pressure_category": "Hypertension Stage 1",
    "bmi": 26.77,
    "bmi_category": "Overweight",
    "cholesterol_category": "Above Normal",
    "glucose_category": "Normal",
    "smoker": false,
    "alcohol": false,
    "physically_active": true
  },
  "recommendations": [
    "Regularly monitor blood pressure, reduce dietary sodium intake (<2,300 mg/day), and consult a doctor.",
    "Current BMI is 26.8 kg/m² (overweight/obese range). Aim for gradual, sustainable caloric balance and fiber-rich meals.",
    "Self-reported cholesterol is above normal. Schedule a lipid panel blood test with your healthcare provider.",
    "Great job staying physically active! Continue maintaining consistent cardiovascular exercise."
  ],
  "message": "Moderate model estimate (58.4%). Some indicators are above baseline ranges.",
  "disclaimer": "Educational Screening Notice: This prediction is generated by an academic machine learning prototype (Group C BCA End-Semester Project) trained on the public Cardiovascular Disease dataset. It is not a clinical diagnosis or medical advice. Individuals should consult a qualified healthcare provider for proper evaluation."
}
```

---

## ⚖️ Functional Requirements Acceptance Matrix

| Requirement ID | Description | Priority | Status | Verification Method |
|:---:|:---|:---:|:---:|:---|
| **FR-01** | Display patient screening form with agreed model input fields. | Must-have | **Completed** | Form renders all 11 demographic & clinical fields with units and tooltips. |
| **FR-02** | Validate submitted values; reject missing, malformed or out-of-range inputs. | Must-have | **Completed** | Client-side HTML5/JS validation + FastAPI Pydantic schema rejection (`systolic > diastolic`, range checks). |
| **FR-03** | Backend exposes API endpoint accepting validated input and returning prediction. | Must-have | **Completed** | `POST /predict` accepts nested and flat JSON, returns structured prediction & metadata. |
| **FR-04** | System loads trained SVC/KNN model and applies same preprocessing pipeline. | Must-have | **Completed** | Standardized `StandardScaler` pipeline loaded via Joblib; produces instant inferences. |
| **FR-05** | Display prediction result with non-diagnostic disclaimer. | Should-have | **Completed** | Dashboard, API responses, and generated PDF reports display explicit medical disclaimers. |

---

## ⚕️ Academic & Ethical Disclaimer

**CardioGuard** is an educational demonstration developed solely for the academic assessment of Group C (BCA 1st Semester). It does **not** provide medical diagnoses, treatment plans, or clinical triage. Machine-learning estimates reflect statistical associations within the public dataset and must not be used as a substitute for professional clinical judgment. Always consult a qualified medical professional for health concerns.
