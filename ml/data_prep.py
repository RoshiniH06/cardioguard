"""
CardioGuard - Data Preparation & Exploratory Data Analysis (EDA)
================================================================
PRD Module Owner: Madhurjya Dewraja (Data Preparation)

This module:
1. Loads the raw 70,000-record Cardiovascular Disease dataset (cardio_train.csv).
2. Performs data quality checks, schema validation, and missing-value inspection.
3. Cleans physiologically implausible outlier values (blood pressure, height, weight, BMI).
4. Performs feature engineering:
   - age_years = age / 365.25
   - bmi = weight / (height / 100) ** 2
5. Generates EDA visualisations:
   - Target class distribution
   - Blood pressure distribution
   - Feature correlation heatmap
6. Exports the prepared, clean dataset for model training:
   - ml/cardio_cleaned.csv
   - ml/outputs/eda_summary.json
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set CPU count for loky to silence Windows warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Set directories
ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent
RAW_DATA_PATH = ML_DIR / "cardio_train.csv"
CLEAN_DATA_PATH = ML_DIR / "cardio_cleaned.csv"
OUTPUTS_DIR = ML_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_inspect_data():
    """Load raw dataset and display baseline summary."""
    print("=" * 65)
    print("STEP 1: LOADING RAW DATASET (cardio_train.csv)")
    print("=" * 65)
    
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_PATH}")
        
    df_raw = pd.read_csv(RAW_DATA_PATH, sep=";")
    print(f"Total raw records : {len(df_raw):,}")
    print(f"Columns ({len(df_raw.columns)}): {list(df_raw.columns)}")
    
    # Missing values check
    missing = df_raw.isna().sum()
    print("\nMissing values per column:")
    for col, count in missing.items():
        print(f"  - {col:<15}: {count}")
        
    return df_raw


def clean_and_engineer_features(df_raw):
    """
    Clean data and engineer clinically relevant features:
    - age in days -> age_years
    - height & weight -> bmi
    - physiological filtering according to CDC/AHA clinical ranges
    """
    print("\n" + "=" * 65)
    print("STEP 2: FEATURE ENGINEERING & DATA CLEANING")
    print("=" * 65)
    
    df = df_raw.copy()
    
    # Feature engineering: Age in years
    df["age_years"] = (df["age"] / 365.25).round(1)
    
    # Feature engineering: Body Mass Index (BMI = kg / m^2)
    df["bmi"] = (df["weight"] / ((df["height"] / 100.0) ** 2)).round(2)
    
    initial_count = len(df)
    
    # Physiological boundary filtering
    # Systolic BP (ap_hi): 60 to 240 mmHg
    # Diastolic BP (ap_lo): 40 to 150 mmHg
    # ap_hi > ap_lo (Systolic must be strictly greater than Diastolic)
    # Height: 120 to 220 cm
    # Weight: 30 to 200 kg
    # BMI: 12 to 60 kg/m^2
    valid_mask = (
        (df["height"] >= 120) & (df["height"] <= 220) &
        (df["weight"] >= 30) & (df["weight"] <= 200) &
        (df["ap_hi"] >= 60) & (df["ap_hi"] <= 240) &
        (df["ap_lo"] >= 40) & (df["ap_lo"] <= 150) &
        (df["ap_hi"] > df["ap_lo"]) &
        (df["bmi"] >= 12) & (df["bmi"] <= 60)
    )
    
    df_clean = df[valid_mask].copy()
    cleaned_count = len(df_clean)
    dropped_count = initial_count - cleaned_count
    
    print(f"Raw records         : {initial_count:,}")
    print(f"Physiological drops : {dropped_count:,} ({dropped_count/initial_count*100:.2f}%)")
    print(f"Cleaned records     : {cleaned_count:,} ({cleaned_count/initial_count*100:.2f}% retained)")
    
    target_counts = df_clean["cardio"].value_counts()
    print("\nTarget Class Distribution:")
    print(f"  - No Disease (0) : {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/cleaned_count*100:.2f}%)")
    print(f"  - Disease (1)    : {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/cleaned_count*100:.2f}%)")
    
    # Save cleaned dataset
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\n[OK] Cleaned dataset saved to: {CLEAN_DATA_PATH}")
    
    return df_clean


def generate_eda_artifacts(df_clean):
    """Generate EDA charts and summary statistics."""
    print("\n" + "=" * 65)
    print("STEP 3: GENERATING EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS")
    print("=" * 65)
    
    # 1. Target Class Distribution Chart
    plt.figure(figsize=(7, 5))
    counts = df_clean["cardio"].value_counts().sort_index()
    bars = plt.bar(["Absence of CVD (0)", "Presence of CVD (1)"], counts, color=["#10b981", "#ef4444"], width=0.55)
    plt.title("Cardiovascular Disease Target Distribution (Cleaned Dataset)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Number of Patients", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h - 3000, f"{h:,}\n({h/len(df_clean)*100:.1f}%)",
                 ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    target_plot_path = OUTPUTS_DIR / "eda_target_distribution.png"
    plt.savefig(target_plot_path, dpi=160)
    plt.close()
    print(f"[OK] Saved target distribution plot: {target_plot_path.name}")
    
    # 2. Blood Pressure Distribution
    plt.figure(figsize=(9, 5))
    plt.hist(df_clean[df_clean["cardio"] == 0]["ap_hi"], bins=30, alpha=0.6, label="No CVD (Systolic)", color="#10b981")
    plt.hist(df_clean[df_clean["cardio"] == 1]["ap_hi"], bins=30, alpha=0.6, label="CVD Present (Systolic)", color="#ef4444")
    plt.axvline(120, color="#f59e0b", linestyle="--", linewidth=1.5, label="Normal Threshold (120 mmHg)")
    plt.axvline(140, color="#dc2626", linestyle="--", linewidth=1.5, label="Hypertension Stage 2 (140 mmHg)")
    plt.title("Systolic Blood Pressure Distribution by CVD Status", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Systolic Blood Pressure (mmHg)", fontsize=11)
    plt.ylabel("Patient Frequency", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    bp_plot_path = OUTPUTS_DIR / "eda_blood_pressure.png"
    plt.savefig(bp_plot_path, dpi=160)
    plt.close()
    print(f"[OK] Saved blood pressure distribution plot: {bp_plot_path.name}")

    # 3. Correlation Heatmap / Bar chart of correlation with target
    features_for_corr = ["age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
                         "cholesterol", "gluc", "smoke", "alco", "active", "bmi"]
    corr_series = df_clean[features_for_corr].apply(lambda x: x.corr(df_clean["cardio"])).sort_values()
    
    plt.figure(figsize=(9, 5.5))
    colors = ["#ef4444" if val > 0 else "#3b82f6" for val in corr_series]
    bars = plt.barh(corr_series.index, corr_series.values, color=colors, height=0.6)
    plt.title("Correlation of Clinical Features with Cardiovascular Disease Target", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Pearson Correlation Coefficient (r)", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        offset = 0.008 if w >= 0 else -0.028
        plt.text(w + offset, bar.get_y() + bar.get_height()/2., f"{w:+.3f}",
                 va="center", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    corr_plot_path = OUTPUTS_DIR / "eda_features_correlation.png"
    plt.savefig(corr_plot_path, dpi=160)
    plt.close()
    print(f"[OK] Saved feature correlation plot: {corr_plot_path.name}")
    
    # 4. Save JSON summary statistics
    summary = {
        "raw_record_count": 70000,
        "clean_record_count": int(len(df_clean)),
        "records_retained_pct": round(len(df_clean) / 70000 * 100, 2),
        "target_balance": {
            "absence_pct": round(float((df_clean["cardio"] == 0).mean() * 100), 2),
            "presence_pct": round(float((df_clean["cardio"] == 1).mean() * 100), 2),
        },
        "feature_means": {
            "average_age_years": round(float(df_clean["age_years"].mean()), 1),
            "average_systolic_bp": round(float(df_clean["ap_hi"].mean()), 1),
            "average_diastolic_bp": round(float(df_clean["ap_lo"].mean()), 1),
            "average_bmi": round(float(df_clean["bmi"].mean()), 1),
            "average_height_cm": round(float(df_clean["height"].mean()), 1),
            "average_weight_kg": round(float(df_clean["weight"].mean()), 1),
        },
        "feature_correlations_with_cardio": {
            k: round(float(v), 4) for k, v in corr_series.items()
        }
    }
    
    summary_path = OUTPUTS_DIR / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[OK] Saved EDA summary JSON: {summary_path.name}")
    
    return summary


def main():
    df_raw = load_and_inspect_data()
    df_clean = clean_and_engineer_features(df_raw)
    generate_eda_artifacts(df_clean)
    print("\n[SUCCESS] Data Preparation & EDA Completed Successfully!")


if __name__ == "__main__":
    main()
