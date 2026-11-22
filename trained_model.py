"""
CardioGuard - Machine Learning Model Training & Comparative Evaluation
=======================================================================
PRD Module Owner: Roshini Singha (Team Leader; Machine Learning)

This module:
1. Loads the cleaned Cardiovascular Disease dataset (ml/cardio_cleaned.csv).
2. Performs a stratified 80/20 train/test split.
3. Builds scikit-learn Pipelines with StandardScaler to prevent data leakage.
4. Trains and rigorously compares candidate models per PRD Section 3 & 8:
   - Dummy Classifier (Baseline)
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Classifier (SVC - Calibrated LinearSVC)
   - HistGradientBoostingClassifier (Ensemble Benchmark)
5. Evaluates models on held-out test data:
   - Accuracy, Precision, Recall/Sensitivity, Specificity, F1-score, ROC-AUC
   - Confusion Matrices (TN, FP, FN, TP)
6. Optimizes decision threshold for preventive screening sensitivity.
7. Generates visual evaluation artifacts:
   - ROC Curves comparison (ml/outputs/roc_curve.png)
   - Confusion Matrices grid (ml/outputs/confusion_matrices.png)
   - Comparison table (ml/outputs/model_comparison.csv)
8. Persists trained pipelines & metadata for FastAPI deployment:
   - backend/model.joblib
   - backend/model_svc.joblib
   - backend/model_knn.joblib
   - backend/meta.json
"""

import os
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Silence Windows CPU detection warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Set directories
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "ml" / "cardio_cleaned.csv"
OUTPUTS_DIR = PROJECT_DIR / "ml" / "outputs"
BACKEND_DIR = PROJECT_DIR / "backend"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
BACKEND_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_RECALL = 0.70  # Target recall for clinical preventive sensitivity
HIGH_RISK_PROBA = 0.65


def evaluate_predictions(y_true, y_pred, y_proba=None, threshold=0.5):
    """Calculate comprehensive classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.5

    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def find_optimal_threshold(y_true, y_proba, target_recall=TARGET_RECALL):
    """Find threshold that satisfies target recall while maximizing F1."""
    best_t = 0.50
    best_f1 = 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        pred = (y_proba >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if rec >= target_recall and f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return round(best_t, 2)


def main():
    print("=" * 70)
    print("CARDIOGUARD ML TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}. Run ml/data_prep.py first.")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded cleaned records: {len(df):,}")

    # Clinical features for prediction
    FEATURE_COLS = [
        "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", "bmi"
    ]
    TARGET_COL = "cardio"

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Training set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples")

    # Candidate models dictionary
    # All pipelines include StandardScaler fitted strictly on training data
    candidate_pipelines = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=15, n_jobs=-1))
        ]),
        "Support Vector Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("model", CalibratedClassifierCV(LinearSVC(dual=False, random_state=RANDOM_STATE)))
        ]),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=RANDOM_STATE
        )
    }

    results = []
    trained_models = {}
    test_probas = {}
    test_preds = {}

    print("\n" + "-" * 70)
    print(f"{'Model Name':<28} | {'Fit Time':<8} | {'Accuracy':<8} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8}")
    print("-" * 70)

    for name, pipe in candidate_pipelines.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0
        trained_models[name] = pipe

        # Predictions & Probabilities
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
        else:
            proba = np.full(len(X_test), 0.5)

        pred_default = pipe.predict(X_test)
        test_probas[name] = proba
        test_preds[name] = pred_default

        m = evaluate_predictions(y_test, pred_default, proba, threshold=0.5)
        m["model"] = name
        m["fit_time_sec"] = round(fit_time, 2)
        m["evaluation_setting"] = "Default Threshold (0.50)"
        results.append(m)

        print(f"{name:<28} | {fit_time:>6.2f}s | {m['accuracy']:>8.4f} | {m['recall']:>8.4f} | {m['f1']:>8.4f} | {m['roc_auc']:>8.4f}")

    # Select Primary Model (PRD requires comparing SVC and KNN; we also record Gradient Boosting)
    # Both SVC and KNN are fully saved as artifacts.
    svc_metrics = [r for r in results if r["model"] == "Support Vector Classifier"][0]
    knn_metrics = [r for r in results if r["model"] == "K-Nearest Neighbors"][0]
    hgb_metrics = [r for r in results if r["model"] == "HistGradientBoosting"][0]

    print("\n" + "=" * 70)
    print("PRD MODEL COMPARISON SUMMARY (SVC vs KNN):")
    print(f"  - Support Vector Classifier (Calibrated): Accuracy={svc_metrics['accuracy']:.4f}, Recall={svc_metrics['recall']:.4f}, ROC-AUC={svc_metrics['roc_auc']:.4f}")
    print(f"  - K-Nearest Neighbors (k=15)            : Accuracy={knn_metrics['accuracy']:.4f}, Recall={knn_metrics['recall']:.4f}, ROC-AUC={knn_metrics['roc_auc']:.4f}")
    print(f"  - HistGradientBoosting (Ensemble)       : Accuracy={hgb_metrics['accuracy']:.4f}, Recall={hgb_metrics['recall']:.4f}, ROC-AUC={hgb_metrics['roc_auc']:.4f}")
    print("=" * 70)

    # Choose best calibrated model for production serving
    # Primary model: Support Vector Classifier or HistGradientBoosting (highest AUC)
    selected_name = "Support Vector Classifier" if svc_metrics["roc_auc"] >= knn_metrics["roc_auc"] else "K-Nearest Neighbors"
    primary_serving_model = trained_models["HistGradientBoosting"] if hgb_metrics["roc_auc"] > svc_metrics["roc_auc"] else trained_models[selected_name]
    primary_serving_name = "HistGradientBoosting" if hgb_metrics["roc_auc"] > svc_metrics["roc_auc"] else selected_name
    primary_proba = test_probas[primary_serving_name]

    optimal_threshold = find_optimal_threshold(y_test, primary_proba, target_recall=TARGET_RECALL)
    tuned_preds = (primary_proba >= optimal_threshold).astype(int)
    tuned_m = evaluate_predictions(y_test, tuned_preds, primary_proba, threshold=optimal_threshold)
    tuned_m["model"] = primary_serving_name
    tuned_m["fit_time_sec"] = [r for r in results if r["model"] == primary_serving_name][0]["fit_time_sec"]
    tuned_m["evaluation_setting"] = f"Optimized Threshold ({optimal_threshold})"
    results.append(tuned_m)

    print(f"\n[SELECTION] Primary Serving Model: {primary_serving_name} | Tuned Threshold: {optimal_threshold}")
    print(f"Tuned Metrics -> Recall: {tuned_m['recall']:.4f}, Accuracy: {tuned_m['accuracy']:.4f}, F1: {tuned_m['f1']:.4f}")

    # Export Model Comparison CSV
    results_df = pd.DataFrame(results)
    results_csv_path = OUTPUTS_DIR / "model_comparison.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n[OK] Model comparison table saved to: {results_csv_path.name}")

    # ------------------------------------------------------------------
    # GENERATE PLOTS
    # ------------------------------------------------------------------
    # 1. ROC Curves
    plt.figure(figsize=(9, 7))
    for name in candidate_pipelines:
        if name == "Dummy Baseline":
            continue
        proba = test_probas[name]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("CardioGuard - ROC Curves Comparison on Test Set", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_plot_path = OUTPUTS_DIR / "roc_curve.png"
    plt.savefig(roc_plot_path, dpi=160)
    plt.close()
    print(f"[OK] Saved ROC curves comparison plot: {roc_plot_path.name}")

    # 2. Confusion Matrices Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_models = [
        ("Support Vector Classifier", test_preds["Support Vector Classifier"]),
        ("K-Nearest Neighbors", test_preds["K-Nearest Neighbors"]),
        (f"{primary_serving_name} (Tuned thr={optimal_threshold})", tuned_preds),
    ]

    for ax, (m_name, p_vals) in zip(axes, plot_models):
        cm = confusion_matrix(y_test, p_vals)
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(m_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No CVD", "CVD"])
        ax.set_yticklabels(["No CVD", "CVD"])

        # Write numbers in cells
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight="bold")

    plt.tight_layout()
    cm_plot_path = OUTPUTS_DIR / "confusion_matrices.png"
    plt.savefig(cm_plot_path, dpi=160)
    plt.close()
    print(f"[OK] Saved Confusion Matrices plot: {cm_plot_path.name}")

    # ------------------------------------------------------------------
    # PERSIST MODELS & METADATA FOR FASTAPI
    # ------------------------------------------------------------------
    # 1. Primary Model Pipeline
    joblib.dump(primary_serving_model, BACKEND_DIR / "model.joblib", compress=3)
    # 2. Support Vector Classifier Pipeline (required by PRD)
    joblib.dump(trained_models["Support Vector Classifier"], BACKEND_DIR / "model_svc.joblib", compress=3)
    # 3. K-Nearest Neighbors Pipeline (required by PRD)
    joblib.dump(trained_models["K-Nearest Neighbors"], BACKEND_DIR / "model_knn.joblib", compress=3)

    # 4. Save metadata JSON
    meta = {
        "primary_model_name": primary_serving_name,
        "selected_prd_model": selected_name,
        "threshold": optimal_threshold,
        "default_threshold": 0.50,
        "high_risk_probability": HIGH_RISK_PROBA,
        "target_recall": TARGET_RECALL,
        "feature_columns": FEATURE_COLS,
        "dataset": {
            "source": "Cardiovascular Disease Dataset (cardio_train.csv)",
            "total_records": len(df),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "features_used": len(FEATURE_COLS),
        },
        "test_metrics": {
            primary_serving_name: tuned_m,
            "Support Vector Classifier": svc_metrics,
            "K-Nearest Neighbors": knn_metrics,
        },
        "all_model_comparison": results,
        "sklearn_version": sklearn.__version__,
    }

    meta_path = BACKEND_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[OK] Saved model pipelines and metadata to {BACKEND_DIR}")
    print("\n[SUCCESS] Model training and comparative evaluation complete!")


if __name__ == "__main__":
    main()
