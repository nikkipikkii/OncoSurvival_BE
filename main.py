import os
import sys
import math
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

# --- Machine Learning Imports ---
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# --- API Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Local Imports (for graphs & genes) ---
from graph import get_hero_graphs
from gene import get_gene_intelligence  # <--- ADDED

# ============================================================
# 1. CORE LOGIC & TRAINING (Exact logic from app.py)
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
    if np.isnan(score): return "Unknown"
    if score >= 0.75: return "High"
    if score >= 0.5: return "Moderate"
    return "Low"

def get_artifacts():
    print("System: Loading artifacts and training models (app.py logic)...")
    
    # --- Paths ---
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models" # Note: check if folder is "Models" or "models" on disk
    CLINICOGENOMIC_DIR = MODELS_DIR / "Clinicogenomic_31genes_v2"
    TCGA_PATH = CLINICOGENOMIC_DIR / "tables" / "tcga_clinicogenomic_31genes_with_surv.csv"
    MB_PATH = CLINICOGENOMIC_DIR / "tables" / "metabric_clinicogenomic_31genes_with_surv.csv"
    SPLITS_DIR = MODELS_DIR / "CoxPH_Final" / "splits"

    if not TCGA_PATH.exists():
        # Fallback for case sensitivity or folder structure differences
        MODELS_DIR = BASE_DIR / "Models"
        SPLITS_DIR = MODELS_DIR / "CoxPH_Final" / "splits"
        CLINICOGENOMIC_DIR = MODELS_DIR / "Clinicogenomic_31genes_v2"
        TCGA_PATH = CLINICOGENOMIC_DIR / "tables" / "tcga_clinicogenomic_31genes_with_surv.csv"
        MB_PATH = CLINICOGENOMIC_DIR / "tables" / "metabric_clinicogenomic_31genes_with_surv.csv"

    if not TCGA_PATH.exists():
        raise FileNotFoundError(f"Data files missing. Checked: {TCGA_PATH}")

    # --- Load Data ---
    df_tcga = pd.read_csv(TCGA_PATH, index_col=0)
    df_mb = pd.read_csv(MB_PATH, index_col=0)

    clinical = ["AGE", "NODE_POS"]
    gene_features = [c for c in df_tcga.columns if c not in ["time", "event", "AGE", "NODE_POS"]]
    features = clinical + gene_features

    # --- Splits ---
    try:
        train_ids = pd.read_csv(SPLITS_DIR / "train_ids.csv").iloc[:, 0]
        test_ids = pd.read_csv(SPLITS_DIR / "test_ids.csv").iloc[:, 0]
        train_ids = train_ids[train_ids.isin(df_tcga.index)]
        test_ids = test_ids[test_ids.isin(df_tcga.index)]
        df_train = df_tcga.loc[train_ids]
        df_test = df_tcga.loc[test_ids]
    except Exception as e:
        print(f"Warning: Split loading failed ({e}). Using full dataset.")
        df_train = df_tcga
        df_test = df_tcga.sample(min(10, len(df_tcga)))

    # --- Scaling (Fit on Train ONLY - Crucial for app.py logic) ---
    scaler = StandardScaler().fit(df_train[features])
    X_train = scaler.transform(df_train[features])
    X_test = scaler.transform(df_test[features])
    X_mb = scaler.transform(df_mb[features])

    # --- DataFrames for Cox ---
    df_train_s = pd.concat([df_train[["time", "event"]], pd.DataFrame(X_train, index=df_train.index, columns=features)], axis=1)
    df_test_s = pd.concat([df_test[["time", "event"]], pd.DataFrame(X_test, index=df_test.index, columns=features)], axis=1)
    df_mb_s = pd.concat([df_mb[["time", "event"]], pd.DataFrame(X_mb, index=df_mb.index, columns=features)], axis=1)

    # --- Train CoxPH ---
    print("System: Training CoxPHFitter...")
    cph = CoxPHFitter()
    cph.fit(df_train_s, "time", "event")

    # --- Train RSF (EXACT app.py parameters) ---
    print("System: Training RSF (n_estimators=500)...")
    y_train = Surv.from_arrays(event=df_train["event"].astype(bool), time=df_train["time"])
    rsf = RandomSurvivalForest(
        n_estimators=500,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=123,
        n_jobs=-1
    )
    rsf.fit(X_train, y_train)

    # --- Pre-calculate Metrics & Risks ---
    risk_test_cox = cph.predict_partial_hazard(df_test_s).values.flatten()
    risk_mb_cox = cph.predict_partial_hazard(df_mb_s).values.flatten()

    ch_test = rsf.predict_cumulative_hazard_function(X_test)
    risk_test_rsf = np.array([fn.y[-1] for fn in ch_test])

    # --- Feature Intelligence Logic (From app.py) ---
    # Cox Importance
    df_imp_cox = cph.params_.to_frame("coef")
    df_imp_cox["feature"] = df_imp_cox.index
    df_imp_cox["abs_coef"] = df_imp_cox["coef"].abs()
    df_imp_cox["type"] = ["clinical" if f in clinical else "gene" for f in df_imp_cox.index]
    df_imp_cox = df_imp_cox.sort_values("abs_coef", ascending=False)

    # RSF Importance
    try:
        df_imp_rsf = pd.DataFrame({
            "feature": features,
            "importance": rsf.feature_importances_,
        })
        df_imp_rsf["abs_importance"] = df_imp_rsf["importance"].abs()
        df_imp_rsf["type"] = ["clinical" if f in clinical else "gene" for f in features]
        df_imp_rsf = df_imp_rsf.sort_values("abs_importance", ascending=False)
    except Exception:
        df_imp_rsf = pd.DataFrame()

    # --- C-Indices (Model Facts) ---
    cindex_test_cox = concordance_index(df_test_s["time"], -risk_test_cox, df_test_s["event"])
    cindex_mb_cox = concordance_index(df_mb_s["time"], -risk_mb_cox, df_mb_s["event"])
    
    # Calculate RSF C-Index (requires censored metric)
    cindex_test_rsf = concordance_index_censored(
        df_test["event"].astype(bool), df_test["time"].astype(float), risk_test_rsf
    )[0]
    
    # Predict RSF risk for MB for C-index calculation
    ch_mb = rsf.predict_cumulative_hazard_function(X_mb)
    risk_mb_rsf = np.array([fn.y[-1] for fn in ch_mb])
    cindex_mb_rsf = concordance_index_censored(
        df_mb["event"].astype(bool), df_mb["time"].astype(float), risk_mb_rsf
    )[0]

    return {
        "scaler": scaler,
        "cph": cph,
        "rsf": rsf,
        "features": features,
        "gene_features": gene_features,
        "clinical": clinical,
        "df_tcga": df_tcga,
        "df_test": df_test_s,
        "df_mb_s": df_mb_s,
        
        # Risk Artifacts
        "risk_mb_cox": risk_mb_cox,
        "risk_test_cox": risk_test_cox,
        "risk_test_rsf": risk_test_rsf,
        
        # Feature Intelligence Artifacts
        "df_imp_cox": df_imp_cox,
        "df_imp_rsf": df_imp_rsf,
        
        # Model Facts / Metrics
        "cindex_test_cox": cindex_test_cox,
        "cindex_mb_cox": cindex_mb_cox,
        "cindex_test_rsf": cindex_test_rsf,
        "cindex_mb_rsf": cindex_mb_rsf,
        
        # Helper for Graphs
        "df_tcga_test_km": df_test_s.assign(risk_cox=risk_test_cox),
        "df_mb_km": df_mb_s.assign(risk_cox=risk_mb_cox)
    }

# ============================================================
# 3. API CONFIGURATION
# ============================================================

app = FastAPI(title="OncoRisk Dual-Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize
print("Initializing Application...")
try:
    art = get_artifacts()
    scaler = art["scaler"]
    cph = art["cph"]
    rsf = art["rsf"]
    features = art["features"]
    gene_features = art["gene_features"]
    risk_mb_cox_global = art["risk_mb_cox"]
    print("Initialization Complete.")
except Exception as e:
    print(f"CRITICAL INITIALIZATION ERROR: {e}")
    traceback.print_exc()

# Models
class InferenceRequest(BaseModel):
    age: float
    nodeStatus: str
    genes: Dict[str, float]

# Helpers
def clean_float(value, precision=2):
    if value is None: return None
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val): return None
        return round(val, precision)
    except: return None

def clean_nan_values(obj):
    if isinstance(obj, float): return clean_float(obj)
    elif isinstance(obj, dict): return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [clean_nan_values(v) for v in obj]
    elif isinstance(obj, (np.float64, np.float32)): return clean_float(obj)
    return obj

# ============================================================
# 4. ENDPOINTS
# ============================================================

@app.get("/metadata")
async def get_metadata():
    df_tcga = art["df_tcga"]
    test_ids = art["df_test"].index.tolist()[:100]
    patients = []
    for pid in test_ids:
        if pid in df_tcga.index:
            patient_genes = df_tcga.loc[pid, gene_features].to_dict()
            patients.append({
                "id": pid,
                "age": float(df_tcga.loc[pid, "AGE"]),
                "node": 1 if df_tcga.loc[pid, "NODE_POS"] == 1 else 0,
                "genes": {k: clean_float(v, 4) for k,v in patient_genes.items()}
            })
    return {"genes": gene_features, "patients": patients}

@app.post("/predict")
async def predict(data: InferenceRequest):
    try:
        # 1. Feature Prep (Robust logic from app.py)
        node_pos = 1 if data.nodeStatus == "Positive" else 0
        row_dict = {"AGE": data.age, "NODE_POS": node_pos}
        row_dict.update(data.genes)
        
        row_raw = pd.DataFrame([row_dict])
        for feature in features:
            if feature not in row_raw.columns:
                row_raw[feature] = 0.0
        
        row_raw = row_raw[features]
        X_row = scaler.transform(row_raw.values)
        df_row_scaled = pd.DataFrame(X_row, columns=features)

        # 2. Inference
        hazard_cox = float(cph.predict_partial_hazard(df_row_scaled).values[0])
        surv_func_cox = cph.predict_survival_function(df_row_scaled)
        times_cox = surv_func_cox.index.values.astype(float)
        surv_cox = surv_func_cox.values[:, 0].astype(float)

        surv_funcs_rsf = rsf.predict_survival_function(X_row)
        sf_rsf = surv_funcs_rsf[0]
        times_rsf = sf_rsf.x.astype(float)
        surv_rsf = sf_rsf.y.astype(float)

        # 3. Percentile & Grid Sync
        percentile = (risk_mb_cox_global < hazard_cox).mean() * 100
        
        t_min = 0.0
        t_max = min(times_cox.max(), times_rsf.max())
        grid = np.linspace(t_min, t_max, 200)

        surv_cox_grid = np.interp(grid, times_cox, surv_cox)
        surv_rsf_grid = np.interp(grid, times_rsf, surv_rsf)

        m_cox = median_survival_time(grid, surv_cox_grid)
        m_rsf = median_survival_time(grid, surv_rsf_grid)
        r_cox = rmst(grid, surv_cox_grid)
        r_rsf = rmst(grid, surv_rsf_grid)
        consensus_median = np.nanmean([m_cox, m_rsf])
        agree = agreement_score(m_cox, m_rsf)

        print(f"Inference -> Age: {data.age}, Node: {data.nodeStatus}, Pct: {percentile:.1f}%")

        return {
            "summary": {
                "coxHazard": clean_float(hazard_cox),
                "rsfRisk": clean_float(float(sf_rsf.y[-1])),
                "agreement": clean_float(agree),
                "agreementLabel": agreement_label(agree),
                "riskPercentile": clean_float(percentile, 1)
            },
            "estimates": {
                "medianCox": clean_float(m_cox, 1),
                "medianRsf": clean_float(m_rsf, 1),
                "consensus": clean_float(consensus_median, 1)
            },
            "rmst": {
                "cox": clean_float(r_cox, 1),
                "rsf": clean_float(r_rsf, 1)
            },
            "curveData": [
                {"time": clean_float(t, 1), "cox": clean_float(c, 4), "rsf": clean_float(r, 4)} 
                for t, c, r in zip(grid, surv_cox_grid, surv_rsf_grid)
            ]
        }
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- UPDATED: Gene Intelligence Endpoint ---
@app.get("/gene-intelligence")
async def gene_intelligence():
    """
    Delegates to gene.py logic for modular narrative handling.
    """
    try:
        data = get_gene_intelligence(art)
        return clean_nan_values(data)
    except Exception as e:
        print(f"Gene Intelligence Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Feature Intelligence Endpoint ---
@app.get("/feature-intelligence")
async def feature_intelligence():
    try:
        # Get Top 30 Cox Features
        df_cox = art["df_imp_cox"].head(30)
        cox_data = [
            {"feature": r["feature"], "value": clean_float(r["coef"], 4), "absVal": clean_float(r["abs_coef"], 4)}
            for _, r in df_cox.iterrows()
        ]

        # Get Top 30 RSF Features
        rsf_data = []
        if not art["df_imp_rsf"].empty:
            df_rsf = art["df_imp_rsf"].head(30)
            rsf_data = [
                {"feature": r["feature"], "value": clean_float(r["importance"], 4)}
                for _, r in df_rsf.iterrows()
            ]

        return {"cox": cox_data, "rsf": rsf_data}
    except Exception as e:
        print(f"Feature Intelligence Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Model Facts Endpoint ---
@app.get("/model-facts")
async def model_facts():
    return {
        "metrics": {
            "cindex_test_cox": clean_float(art["cindex_test_cox"], 3),
            "cindex_mb_cox": clean_float(art["cindex_mb_cox"], 3),
            "cindex_test_rsf": clean_float(art["cindex_test_rsf"], 3),
            "cindex_mb_rsf": clean_float(art["cindex_mb_rsf"], 3)
        },
        "info": {
            "tcga_count": len(art["df_tcga"]),
            "metabric_count": len(art["df_mb_s"]),
            "feature_count": len(art["features"])
        }
    }

# --- UPDATED: Hero Graphs Endpoint ---
@app.get("/hero-graphs")
async def hero_graphs_endpoint():
    """
    Delegates to graph.py logic.
    """
    try:
        data = get_hero_graphs(art)
        return clean_nan_values(data)
    except Exception as e:
        print(f"Hero Graphs Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
