import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# ============================================================
# PATHS
# ============================================================
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
    print("model_logic: Loading artifacts...")

    # 1. Load Data
    if not TCGA_PATH.exists() or not MB_PATH.exists():
        raise FileNotFoundError(f"Data files missing.")

    df_tcga = pd.read_csv(TCGA_PATH, index_col=0)
    df_mb = pd.read_csv(MB_PATH, index_col=0)

    # 2. Define Features
    clinical = ["AGE", "NODE_POS"]
    gene_features = [c for c in df_tcga.columns if c not in ["time", "event", "AGE", "NODE_POS"]]
    features = clinical + gene_features

    # 3. Load Splits
    train_ids = pd.read_csv(SPLITS_DIR / "train_ids.csv").iloc[:, 0]
    test_ids = pd.read_csv(SPLITS_DIR / "test_ids.csv").iloc[:, 0]
    
    train_ids = train_ids[train_ids.isin(df_tcga.index)]
    test_ids = test_ids[test_ids.isin(df_tcga.index)]

    df_train = df_tcga.loc[train_ids]
    df_test = df_tcga.loc[test_ids]

    # 4. Scale Data
    scaler = StandardScaler().fit(df_train[features])
    X_train = scaler.transform(df_train[features])
    X_mb = scaler.transform(df_mb[features])
    
    # We need X_test for calculations
    X_test = scaler.transform(df_test[features])

    # DataFrames for Cox
    df_train_s = pd.concat([df_train[["time", "event"]], pd.DataFrame(X_train, index=df_train.index, columns=features)], axis=1)
    df_test_s = pd.concat([df_test[["time", "event"]], pd.DataFrame(X_test, index=df_test.index, columns=features)], axis=1)
    df_mb_s = pd.concat([df_mb[["time", "event"]], pd.DataFrame(X_mb, index=df_mb.index, columns=features)], axis=1)

    # 5. Train Models
    print("model_logic: Training Cox...")
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df_train_s, duration_col="time", event_col="event")

    print("model_logic: Training RSF...")
    y_train = Surv.from_arrays(event=df_train["event"].astype(bool), time=df_train["time"])
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=42, n_jobs=-1)
    rsf.fit(X_train, y_train)

    # 6. CALCULATE MISSING METRICS (The Fix for KeyError)
    print("model_logic: Calculating metrics for graphs...")
    
    # Risk Scores
    risk_test_cox = cph.predict_partial_hazard(df_test_s).values.flatten()
    risk_mb_cox = cph.predict_partial_hazard(df_mb_s).values.flatten()
    
    # RSF Risk Scores (Cumulative Hazard)
    ch_test = rsf.predict_cumulative_hazard_function(X_test)
    risk_test_rsf = np.array([fn.y[-1] for fn in ch_test])

    # KM Grouping
    df_tcga_test_km = df_test_s.copy()
    df_tcga_test_km["risk_cox"] = risk_test_cox
    cut_tcga = np.median(risk_test_cox)
    df_tcga_test_km["group_cox"] = np.where(df_tcga_test_km["risk_cox"] > cut_tcga, "High", "Low")

    df_mb_km = df_mb_s.copy()
    df_mb_km["risk_cox"] = risk_mb_cox
    cut_mb = np.median(risk_mb_cox)
    df_mb_km["group_cox"] = np.where(df_mb_km["risk_cox"] > cut_mb, "High", "Low")

    # Feature Importance (Cox)
    df_imp_cox = cph.params_.to_frame("coef")
    df_imp_cox["feature"] = df_imp_cox.index
    df_imp_cox["abs_coef"] = df_imp_cox["coef"].abs()
    df_imp_cox["type"] = ["clinical" if f in clinical else "gene" for f in df_imp_cox.index]
    df_imp_cox = df_imp_cox.sort_values("abs_coef", ascending=False)

    # C-Indices (Optional but good for completeness)
    cindex_test_cox = concordance_index(df_test_s["time"], -risk_test_cox, df_test_s["event"])
    cindex_mb_cox = concordance_index(df_mb_s["time"], -risk_mb_cox, df_mb_s["event"])

    print("model_logic: Success.")

    return {
        "scaler": scaler,
        "cph": cph,
        "rsf": rsf,
        "features": features,
        "gene_features": gene_features,
        "df_tcga": df_tcga,
        "df_test": df_test_s,
        "df_mb_s": df_mb_s,
        "clinical": clinical,
        
        # --- KEYS REQUIRED BY graph.py ---
        "df_tcga_test_km": df_tcga_test_km,
        "df_mb_km": df_mb_km,
        "risk_test_cox": risk_test_cox,
        "risk_test_rsf": risk_test_rsf,
        "df_imp_cox": df_imp_cox,
        "cindex_test_cox": cindex_test_cox,
        "cindex_mb_cox": cindex_mb_cox
    }
