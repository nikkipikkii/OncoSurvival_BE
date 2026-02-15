import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# ============================================================
# PATHS
# ============================================================
# Fix for Render: Ensure 'models' is lowercase if your repo uses lowercase
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SPLITS_DIR = MODELS_DIR / "CoxPH_Final" / "splits"
CLINICOGENOMIC_DIR = MODELS_DIR / "Clinicogenomic_31genes_v2"
TCGA_PATH = CLINICOGENOMIC_DIR / "tables" / "tcga_clinicogenomic_31genes_with_surv.csv"
MB_PATH = CLINICOGENOMIC_DIR / "tables" / "metabric_clinicogenomic_31genes_with_surv.csv"

# ============================================================
# LOGIC FUNCTIONS
# ============================================================
def median_survival_time(times, surv):
    below = surv <= 0.5
    if not np.any(below):
        return np.nan
    return float(times[np.argmax(below)])

def rmst(times, surv):
    return float(np.trapz(surv, times))

def agreement_score(median_cox, median_rsf):
    if np.isnan(median_cox) or np.isnan(median_rsf):
        return np.nan
    denom = max(median_cox, median_rsf)
    if denom == 0:
        return np.nan
    delta = abs(median_cox - median_rsf) / denom
    return float(max(0.0, min(1.0, 1.0 - delta)))

def agreement_label(score):
    if np.isnan(score):
        return "Unknown"
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Moderate"
    return "Low"

def get_artifacts():
    """Loads and caches models and data."""
    print("model_logic: STARTING artifact loading...")

    # 1. Check Paths
    if not TCGA_PATH.exists():
        print(f"model_logic: ERROR - TCGA File not found at {TCGA_PATH}")
        raise FileNotFoundError(f"TCGA file missing: {TCGA_PATH}")
    if not MB_PATH.exists():
        print(f"model_logic: ERROR - MB File not found at {MB_PATH}")
        raise FileNotFoundError(f"MB file missing: {MB_PATH}")

    # 2. Load Data
    print("model_logic: Reading CSV files...")
    df_tcga = pd.read_csv(TCGA_PATH, index_col=0)
    df_mb = pd.read_csv(MB_PATH, index_col=0)
    print(f"model_logic: TCGA Data Loaded. Shape: {df_tcga.shape}")
    print(f"model_logic: METABRIC Data Loaded. Shape: {df_mb.shape}")

    # 3. Define Features
    clinical = ["AGE", "NODE_POS"]
    gene_features = [
        c for c in df_tcga.columns
        if c not in ["time", "event", "AGE", "NODE_POS"]
    ]
    features = clinical + gene_features
    print(f"model_logic: Features defined. Total count: {len(features)}")

    # 4. Load Splits
    print("model_logic: Loading train/test splits...")
    if not (SPLITS_DIR / "train_ids.csv").exists():
         print(f"model_logic: ERROR - Split files missing at {SPLITS_DIR}")
         
    train_ids = pd.read_csv(SPLITS_DIR / "train_ids.csv").iloc[:, 0]
    test_ids = pd.read_csv(SPLITS_DIR / "test_ids.csv").iloc[:, 0]
    
    # Filter IDs to ensure they exist in dataframe
    train_ids = train_ids[train_ids.isin(df_tcga.index)]
    test_ids = test_ids[test_ids.isin(df_tcga.index)]
    print(f"model_logic: Split sizes - Train: {len(train_ids)}, Test: {len(test_ids)}")

    df_train = df_tcga.loc[train_ids]
    df_test = df_tcga.loc[test_ids]

    # 5. Fit Scaler
    print("model_logic: Fitting Standard Scaler...")
    scaler = StandardScaler().fit(df_train[features])
    
    # Transform Data
    X_train = scaler.transform(df_train[features])
    X_mb = scaler.transform(df_mb[features])
    # Note: We don't transform df_test here for the global object, 
    # but we will need it for the KM plots below.

    # Prepare DataFrames for Training
    df_train_s = pd.concat(
        [df_train[["time", "event"]],
         pd.DataFrame(X_train, index=df_train.index, columns=features)], axis=1
    )
    
    df_mb_s = pd.concat(
        [df_mb[["time", "event"]],
         pd.DataFrame(X_mb, index=df_mb.index, columns=features)], axis=1
    )

    # 6. Train Cox Model
    print("model_logic: Training CoxPH Model...")
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df_train_s, duration_col="time", event_col="event")
    print("model_logic: CoxPH Training Complete.")

    # 7. Train RSF Model
    print("model_logic: Training Random Survival Forest (this may take 10-20 seconds)...")
    y_train = Surv.from_arrays(
        event=df_train["event"].astype(bool),
        time=df_train["time"]
    )
    rsf = RandomSurvivalForest(
        n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=42, n_jobs=-1
    )
    rsf.fit(X_train, y_train)
    print("model_logic: RSF Training Complete.")

    # 8. Generate KM Data for Hero Graphs (THE FIX FOR KEYERROR)
    print("model_logic: Generating KM Dataframes for Hero Graphs...")
    
    # TCGA Test Set KM Data
    # We need to scale the test set first to get risk scores
    X_test_for_km = scaler.transform(df_test[features])
    df_test_scaled_for_km = pd.DataFrame(X_test_for_km, index=df_test.index, columns=features)
    
    risk_test_cox = cph.predict_partial_hazard(df_test_scaled_for_km).values.flatten()
    df_tcga_test_km = df_test.copy()
    df_tcga_test_km["risk_cox"] = risk_test_cox
    cut_tcga = np.median(risk_test_cox)
    df_tcga_test_km["group_cox"] = np.where(df_tcga_test_km["risk_cox"] > cut_tcga, "High", "Low")
    print(f"model_logic: df_tcga_test_km created. Columns: {df_tcga_test_km.columns.tolist()}")

    # METABRIC KM Data
    risk_mb_cox = cph.predict_partial_hazard(df_mb_s).values.flatten()
    df_mb_km = df_mb_s.copy()
    df_mb_km["risk_cox"] = risk_mb_cox
    cut_mb = np.median(risk_mb_cox)
    df_mb_km["group_cox"] = np.where(df_mb_km["risk_cox"] > cut_mb, "High", "Low")
    print(f"model_logic: df_mb_km created. Columns: {df_mb_km.columns.tolist()}")

    print("model_logic: Artifact loading SUCCESSFUL. Returning dictionary.")
    
    return {
        "scaler": scaler,
        "cph": cph,
        "rsf": rsf,
        "features": features,
        "gene_features": gene_features,
        "df_tcga": df_tcga,
        "df_test": df_test,
        "df_mb_s": df_mb_s,
        "clinical": clinical,
        # THESE are the keys that were missing causing your 500 Error
        "df_tcga_test_km": df_tcga_test_km,
        "df_mb_km": df_mb_km
    }
