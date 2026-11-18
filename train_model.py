"""
CardioGuard - Training on BRFSS-derived CVD dataset (CVD_cleaned.csv)
=====================================================================
Run:  python train_model.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_RECALL = 0.50  # Changed to 0.50 so it doesn't break when optimizing for accuracy
HIGH_RISK_PROBA = 0.65

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

DATA_PATH = PROJECT_DIR / "ml" / "CVD_cleaned.csv"
OUT_DIR = PROJECT_DIR / "ml" / "outputs"
BACKEND_DIR = PROJECT_DIR / "backend"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BACKEND_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load + INSPECT
# ------------------------------------------------------------------
raw = pd.read_csv(DATA_PATH)
print("Shape:", raw.shape)
print("\nMissing values:\n", raw.isna().sum()[raw.isna().sum() > 0])

TARGET_COL = "Heart_Disease"
print(f"\nUsing target column: {TARGET_COL!r}")
print("Target values:\n", raw[TARGET_COL].value_counts())

# ------------------------------------------------------------------
# 2. Feature column definitions
# ------------------------------------------------------------------
NUM_COLS = ["BMI", "Alcohol_Consumption", "Fruit_Consumption", 
            "Green_Vegetables_Consumption", "FriedPotato_Consumption",
            "Height_(cm)", "Weight_(kg)"]
            
CAT_COLS = ["General_Health", "Checkup", "Exercise", "Diabetes",
            "Depression", "Age_Category", "Sex", "Smoking_History",
            "Skin_Cancer", "Other_Cancer", "Arthritis"]

BIN_COLS = []
FEATURE_COLS = NUM_COLS + CAT_COLS + BIN_COLS

# Defensive check to ensure columns exist
present = set(raw.columns)
NUM_COLS = [c for c in NUM_COLS if c in present]
CAT_COLS = [c for c in CAT_COLS if c in present]
FEATURE_COLS = NUM_COLS + CAT_COLS + BIN_COLS

# ------------------------------------------------------------------
# 3. Clean the target and basic filtering
# ------------------------------------------------------------------
df = raw.copy()

# Map Yes/No to 1/0
df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
df = df.dropna(subset=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(int)

# Drop rows missing any feature
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

# Basic numeric sanity (BMI range etc.)
if "BMI" in df.columns:
    df = df[(df["BMI"] > 10) & (df["BMI"] < 70)]

print(f"\nRows kept after cleaning: {len(df)}")
print("Target balance:\n", df[TARGET_COL].value_counts(normalize=True).round(3))

# ------------------------------------------------------------------
# 4. Stratified 70/15/15 split
# ------------------------------------------------------------------
X = df[FEATURE_COLS]
y = df[TARGET_COL]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15 / 0.85, stratify=y_trainval, random_state=RANDOM_STATE
)
print("\nTrain:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# ------------------------------------------------------------------
# 5. Preprocessing + models (Class weights REMOVED)
# ------------------------------------------------------------------
def make_preprocessor(scale=True):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("bin", "passthrough", BIN_COLS),
    ])

def make_pipeline(model, scale=True):
    return Pipeline([("pre", make_preprocessor(scale)), ("model", model)])

def evaluate(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

def pick_threshold(y_true, proba, target_recall=TARGET_RECALL):
    best = None
    for t in np.arange(0.10, 0.91, 0.01):
        if recall_score(y_true, (proba >= t).astype(int), zero_division=0) >= target_recall:
            best = float(t)
    return best if best is not None else 0.30

models = {
    "LogReg": make_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        scale=True,
    ),
    "RandomForest": make_pipeline(
        RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=20,
            n_jobs=-1, random_state=RANDOM_STATE),
        scale=False,
    ),
    "HistGradientBoosting": make_pipeline(
        HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=RANDOM_STATE),
        scale=False,
    ),
}

# ------------------------------------------------------------------
# 6. Train + evaluate
# ------------------------------------------------------------------
val_proba, test_proba = {}, {}
print("\n--- Validation ---")
for name, m in models.items():
    m.fit(X_train, y_train)
    val_proba[name] = m.predict_proba(X_val)[:, 1]
    test_proba[name] = m.predict_proba(X_test)[:, 1]
    print(f"{name:22s}  val acc {accuracy_score(y_val, val_proba[name] >= 0.5):.4f}"
          f"  val AUC {roc_auc_score(y_val, val_proba[name]):.4f}")

# Ensemble
ens_val = np.mean([val_proba[n] for n in models], axis=0)
ens_test = np.mean([test_proba[n] for n in models], axis=0)
val_proba["Ensemble"] = ens_val
test_proba["Ensemble"] = ens_test
print(f"{'Ensemble':22s}  val acc {accuracy_score(y_val, ens_val >= 0.5):.4f}"
      f"  val AUC {roc_auc_score(y_val, ens_val):.4f}")

# ------------------------------------------------------------------
# 7. Choose model by VALIDATION AUC
# ------------------------------------------------------------------
best_name = max(val_proba, key=lambda n: roc_auc_score(y_val, val_proba[n]))
thresholds = {n: pick_threshold(y_val, val_proba[n]) for n in val_proba}
best_threshold = thresholds[best_name]
print(f"\nBest by val AUC: {best_name} | chosen threshold: {best_threshold:.2f}")

# ------------------------------------------------------------------
# 8. Final test metrics
# ------------------------------------------------------------------
rows = []
for name in val_proba:
    for setting, thr in [("default 0.50", 0.5), ("tuned", thresholds[name])]:
        m = evaluate(y_test, test_proba[name], thr)
        rows.append({"model": name, "setting": setting,
                     **{k: m[k] for k in ["threshold", "accuracy", "precision",
                                           "recall", "f1", "roc_auc", "FN"]}})
results_df = pd.DataFrame(rows)
print("\n--- Test set ---")
print(results_df.round(3).to_string(index=False))
results_df.round(4).to_csv(OUT_DIR / "model_comparison.csv", index=False)

best_test = evaluate(y_test, test_proba[best_name], best_threshold)

# ------------------------------------------------------------------
# 9. Save model + metadata for the API
# ------------------------------------------------------------------
if best_name == "Ensemble":
    saved_model = models["HistGradientBoosting"]
    saved_name = "HistGradientBoosting"
else:
    saved_model = models[best_name]
    saved_name = best_name

joblib.dump(saved_model, BACKEND_DIR / "model.joblib", compress=3)

meta = {
    "model_name": saved_name,
    "threshold": round(best_threshold, 2),
    "high_risk_probability": HIGH_RISK_PROBA,
    "target_recall": TARGET_RECALL,
    "feature_columns": FEATURE_COLS,
    "num_cols": NUM_COLS,
    "cat_cols": CAT_COLS,
    "test_metrics": {k: round(float(best_test[k]), 4)
                     for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
    "rows_used": int(len(df)),
    "sklearn_version": sklearn.__version__,
}
(BACKEND_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nSaved model + meta to {BACKEND_DIR}")

# ------------------------------------------------------------------
# 10. ROC curve chart
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
for name in val_proba:
    fpr, tpr, _ = roc_curve(y_test, test_proba[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, test_proba[name]):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve — test set")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curve.png", dpi=150)
plt.show()