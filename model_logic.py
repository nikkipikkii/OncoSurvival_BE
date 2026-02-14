import os
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
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SPLITS_DIR = MODELS_DIR / "CoxPH_Final" / "splits"
CLINICOGENOMIC_DIR = MODELS_DIR / "Clinicogenomic_31genes_v2"
TCGA_PATH = CLINICOGENOMIC_DIR / "tables" / "tcga_clinicogenomic_31genes_with_surv.csv"
MB_PATH = CLINICOGENOMIC_DIR / "tables" / "metabric_clinicogenomic_31genes_with_surv.csv"

# ============================================================
# LOGIC FUNCTIONS (Pure Python, No Streamlit)
# ============================================================
def median_survival_time(times, surv):
    """Calculates median survival time where Survival probability <= 0.5."""
    below = surv <= 0.5
    if not np.any(below):
        return np.nan
    return float(times[np.argmax(below)])

def rmst(times, surv):
    """Restricted Mean Survival Time (Area under the curve)."""
    return float(np.trapz(surv, times))

def agreement_score(median_cox, median_rsf):
    """Calculates agreement between Cox and RSF models."""
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
    print("Loading artifacts...")
    
    # Load Data
    if not TCGA_PATH.exists() or not MB_PATH.exists():
        raise FileNotFoundError(f"Data files not found at {TCGA_PATH} or {MB_PATH}")

    df_tcga = pd.read_csv(TCGA_PATH, index_col=0)
    df_mb = pd.read_csv(MB_PATH, index_col=0)

    # Define Features
    clinical = ["AGE", "NODE_POS"]
    gene_features = [
        c for c in df_tcga.columns
        if c not in ["time", "event", "AGE", "NODE_POS"]
    ]
    features = clinical + gene_features

    # Load Splits
    train_ids = pd.read_csv(SPLITS_DIR / "train_ids.csv").iloc[:, 0]
    test_ids = pd.read_csv(SPLITS_DIR / "test_ids.csv").iloc[:, 0]

    # Filter IDs to ensure they exist in the dataframe
    train_ids = train_ids[train_ids.isin(df_tcga.index)]
    test_ids = test_ids[test_ids.isin(df_tcga.index)]

    df_train = df_tcga.loc[train_ids]
    df_test = df_tcga.loc[test_ids]

    # Fit Scaler
    scaler = StandardScaler().fit(df_train[features])

    # Transform Data
    X_train = scaler.transform(df_train[features])
    X_mb = scaler.transform(df_mb[features])

    # Prepare DataFrames for Models
    df_train_s = pd.concat(
        [df_train[["time", "event"]],
         pd.DataFrame(X_train, index=df_train.index, columns=features)], axis=1
    )
    
    df_mb_s = pd.concat(
        [df_mb[["time", "event"]],
         pd.DataFrame(X_mb, index=df_mb.index, columns=features)], axis=1
    )

    # Train/Fit Models (In production, you might want to load saved models instead of retraining)
    # CoxPH
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df_train_s, duration_col="time", event_col="event")

    # RSF
    y_train = Surv.from_arrays(
        event=df_train["event"].astype(bool),
        time=df_train["time"]
    )
    rsf = RandomSurvivalForest(
        n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=42
    )
    rsf.fit(X_train, y_train)

    return {
        "scaler": scaler,
        "cph": cph,
        "rsf": rsf,
        "features": features,
        "gene_features": gene_features,
        "df_tcga": df_tcga,
        "df_test": df_test,
        "df_mb_s": df_mb_s,
        "clinical": clinical
    }
