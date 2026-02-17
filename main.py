import os
import sys
import math
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

# --- Machine Learning Imports (Required for Type Handling) ---
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest

# --- API Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LOCAL MODULE IMPORTS ---
# 1. Graph and Gene logic preserved from external files
from graph import get_hero_graphs
from gene import get_gene_intelligence

# 2. Only import artifact loading. 
# WE DO NOT import math helpers; we define them locally below.
from model_logic import get_artifacts

# ============================================================
# 1. MATH & LOGIC HELPERS (Restored locally from v2)
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

# ============================================================
# 2. INITIALIZATION & ARTIFACT LOADING
# ============================================================

print("Initializing Application & Loading Artifacts...")
try:
    # Load all models and data from the external module
    art = get_artifacts()
    
    # Unpack core components for the /predict endpoint
    scaler = art["scaler"]
    cph = art["cph"]
    rsf = art["rsf"]
    features = art["features"]
    gene_features = art["gene_features"]
    
    # [CRITICAL FIX] 
    # The v2 predict logic requires 'risk_mb_cox_global' as a numpy array.
    # In model_logic.py, this is stored inside the 'df_mb_km' dataframe.
    # We extract it here so the v2 code below runs without modification.
    risk_mb_cox_global = art["df_mb_km"]["risk_cox"].values
    
    print("Initialization Complete. Models Ready.")
except Exception as e:
    print(f"CRITICAL INITIALIZATION ERROR: {e}")
    traceback.print_exc()

# ============================================================
# 3. API CONFIGURATION
# ============================================================

app = FastAPI(title="OncoRisk Dual-Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 4. DATA MODELS & JSON HELPERS
# ============================================================

class InferenceRequest(BaseModel):
    age: float
    nodeStatus: str
    genes: Dict[str, float]

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
# 5. ENDPOINTS
# ============================================================

@app.get("/")
def health_check():
    return {"status": "running", "models_loaded": art is not None}

@app.get("/metadata")
async def get_metadata():
    """
    Returns patient list and gene list (Hybrid Logic).
    """
    try:
        df_tcga = art["df_tcga"]
        # Use first 100 IDs for the frontend dropdown
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
    except Exception as e:
        print(f"Metadata Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(data: InferenceRequest):
    """
    CORE INFERENCE LOGIC - V2 (STRICTLY PRESERVED)
    Uses local math helpers defined above.
    """
    try:
        # --- 1. FEATURE PREPARATION (V2) ---
        node_pos = 1 if data.nodeStatus == "Positive" else 0
        row_dict = {"AGE": data.age, "NODE_POS": node_pos}
        row_dict.update(data.genes)
        
        row_raw = pd.DataFrame([row_dict])
        # Ensure all features exist and are in correct order
        for feature in features:
            if feature not in row_raw.columns:
                row_raw[feature] = 0.0
        
        row_raw = row_raw[features]
        X_row = scaler.transform(row_raw.values)
        df_row_scaled = pd.DataFrame(X_row, columns=features)

        # --- 2. INFERENCE (V2) ---
        # Cox
        hazard_cox = float(cph.predict_partial_hazard(df_row_scaled).values[0])
        surv_func_cox = cph.predict_survival_function(df_row_scaled)
        times_cox = surv_func_cox.index.values.astype(float)
        surv_cox = surv_func_cox.values[:, 0].astype(float)

        # RSF
        surv_funcs_rsf = rsf.predict_survival_function(X_row)
        sf_rsf = surv_funcs_rsf[0]
        times_rsf = sf_rsf.x.astype(float)
        surv_rsf = sf_rsf.y.astype(float)

        # --- 3. PERCENTILE & GRID SYNC (V2) ---
        # Uses 'risk_mb_cox_global' extracted during Init
        percentile = (risk_mb_cox_global < hazard_cox).mean() * 100
        
        t_min = 0.0
        t_max = min(times_cox.max(), times_rsf.max())
        grid = np.linspace(t_min, t_max, 200)

        surv_cox_grid = np.interp(grid, times_cox, surv_cox)
        surv_rsf_grid = np.interp(grid, times_rsf, surv_rsf)

        # Uses LOCAL math helpers (not imported)
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
        print(f"Prediction Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gene-intelligence")
async def gene_intelligence():
    """
    Delegates to gene.py logic (Preserved)
    """
    try:
        data = get_gene_intelligence(art)
        return clean_nan_values(data)
    except Exception as e:
        print(f"Gene Intelligence Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hero-graphs")
async def hero_graphs_endpoint():
    """
    Delegates to graph.py logic (Preserved).
    """
    try:
        data = get_hero_graphs(art)
        return clean_nan_values(data)
    except Exception as e:
        print(f"Hero Graphs Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-intelligence")
async def feature_intelligence():
    try:
        df_cox = art["df_imp_cox"].head(30)
        cox_data = [
            {"feature": r["feature"], "value": clean_float(r["coef"], 4), "absVal": clean_float(r["abs_coef"], 4)}
            for _, r in df_cox.iterrows()
        ]
        
        rsf_data = []
        if "df_imp_rsf" in art and not art["df_imp_rsf"].empty:
             df_rsf = art["df_imp_rsf"].head(30)
             rsf_data = [
                {"feature": r["feature"], "value": clean_float(r["importance"], 4)}
                for _, r in df_rsf.iterrows()
            ]
        
        return {"cox": cox_data, "rsf": rsf_data}
    except Exception as e:
        print(f"Feature Intelligence Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


# import os
# import sys
# import math
# import traceback
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Optional, Any

# # --- API Imports ---
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# # --- LOCAL MODULE IMPORTS (Restored from v1) ---
# # This ensures we use the correct external logic for graphs and genes
# from graph import get_hero_graphs
# from gene import get_gene_intelligence
# from model_logic import (
#     get_artifacts,
#     median_survival_time,
#     rmst,
#     agreement_score,
#     agreement_label
# )

# # ============================================================
# # 1. API CONFIGURATION
# # ============================================================

# app = FastAPI(title="OncoRisk Dual-Model API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all for dev; restrict in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================================================
# # 2. INITIALIZATION (Using model_logic.py)
# # ============================================================

# print("Initializing Application & Loading Artifacts...")
# try:
#     # Load all models and data from the external module
#     art = get_artifacts()
    
#     # Unpack core components for the /predict endpoint
#     scaler = art["scaler"]
#     cph = art["cph"]
#     rsf = art["rsf"]
#     features = art["features"]
#     gene_features = art["gene_features"]
    
#     # DETECTIVE NOTE: model_logic.py packs 'risk_mb_cox' inside 'df_mb_km'.
#     # We extract it here so the /predict logic (v2) works without modification.
#     risk_mb_cox_global = art["df_mb_km"]["risk_cox"].values
    
#     print("Initialization Complete. Models Ready.")
# except Exception as e:
#     print(f"CRITICAL INITIALIZATION ERROR: {e}")
#     traceback.print_exc()

# # ============================================================
# # 3. HELPERS & MODELS (From v2 - "Best Version")
# # ============================================================

# class InferenceRequest(BaseModel):
#     age: float
#     nodeStatus: str
#     genes: Dict[str, float]

# def clean_float(value, precision=2):
#     if value is None: return None
#     try:
#         val = float(value)
#         if math.isnan(val) or math.isinf(val): return None
#         return round(val, precision)
#     except: return None

# def clean_nan_values(obj):
#     if isinstance(obj, float): return clean_float(obj)
#     elif isinstance(obj, dict): return {k: clean_nan_values(v) for k, v in obj.items()}
#     elif isinstance(obj, list): return [clean_nan_values(v) for v in obj]
#     elif isinstance(obj, (np.float64, np.float32)): return clean_float(obj)
#     return obj

# # ============================================================
# # 4. ENDPOINTS
# # ============================================================

# @app.get("/")
# def health_check():
#     return {"status": "running", "models_loaded": art is not None}

# @app.get("/metadata")
# async def get_metadata():
#     """
#     Returns patient list and gene list (Hybrid v1/v2 logic).
#     """
#     try:
#         df_tcga = art["df_tcga"]
#         # Use first 100 IDs for the frontend dropdown
#         test_ids = art["df_test"].index.tolist()[:100]
        
#         patients = []
#         for pid in test_ids:
#             if pid in df_tcga.index:
#                 patient_genes = df_tcga.loc[pid, gene_features].to_dict()
#                 patients.append({
#                     "id": pid,
#                     "age": float(df_tcga.loc[pid, "AGE"]),
#                     "node": 1 if df_tcga.loc[pid, "NODE_POS"] == 1 else 0,
#                     "genes": {k: clean_float(v, 4) for k,v in patient_genes.items()}
#                 })
#         return {"genes": gene_features, "patients": patients}
#     except Exception as e:
#         print(f"Metadata Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/predict")
# async def predict(data: InferenceRequest):
#     """
#     CORE INFERENCE LOGIC - V2 (PRESERVED EXACTLY AS REQUESTED)
#     """
#     try:
#         # 1. Feature Prep
#         node_pos = 1 if data.nodeStatus == "Positive" else 0
#         row_dict = {"AGE": data.age, "NODE_POS": node_pos}
#         row_dict.update(data.genes)
        
#         row_raw = pd.DataFrame([row_dict])
#         # Ensure all features exist and are in correct order
#         for feature in features:
#             if feature not in row_raw.columns:
#                 row_raw[feature] = 0.0
        
#         row_raw = row_raw[features]
#         X_row = scaler.transform(row_raw.values)
#         df_row_scaled = pd.DataFrame(X_row, columns=features)

#         # 2. Inference
#         # Cox
#         hazard_cox = float(cph.predict_partial_hazard(df_row_scaled).values[0])
#         surv_func_cox = cph.predict_survival_function(df_row_scaled)
#         times_cox = surv_func_cox.index.values.astype(float)
#         surv_cox = surv_func_cox.values[:, 0].astype(float)

#         # RSF
#         surv_funcs_rsf = rsf.predict_survival_function(X_row)
#         sf_rsf = surv_funcs_rsf[0]
#         times_rsf = sf_rsf.x.astype(float)
#         surv_rsf = sf_rsf.y.astype(float)

#         # 3. Percentile & Grid Sync
#         # Note: Using risk_mb_cox_global extracted during Init
#         percentile = (risk_mb_cox_global < hazard_cox).mean() * 100
        
#         t_min = 0.0
#         t_max = min(times_cox.max(), times_rsf.max())
#         grid = np.linspace(t_min, t_max, 200)

#         surv_cox_grid = np.interp(grid, times_cox, surv_cox)
#         surv_rsf_grid = np.interp(grid, times_rsf, surv_rsf)

#         m_cox = median_survival_time(grid, surv_cox_grid)
#         m_rsf = median_survival_time(grid, surv_rsf_grid)
#         r_cox = rmst(grid, surv_cox_grid)
#         r_rsf = rmst(grid, surv_rsf_grid)
#         consensus_median = np.nanmean([m_cox, m_rsf])
#         agree = agreement_score(m_cox, m_rsf)

#         print(f"Inference -> Age: {data.age}, Node: {data.nodeStatus}, Pct: {percentile:.1f}%")

#         return {
#             "summary": {
#                 "coxHazard": clean_float(hazard_cox),
#                 "rsfRisk": clean_float(float(sf_rsf.y[-1])),
#                 "agreement": clean_float(agree),
#                 "agreementLabel": agreement_label(agree),
#                 "riskPercentile": clean_float(percentile, 1)
#             },
#             "estimates": {
#                 "medianCox": clean_float(m_cox, 1),
#                 "medianRsf": clean_float(m_rsf, 1),
#                 "consensus": clean_float(consensus_median, 1)
#             },
#             "rmst": {
#                 "cox": clean_float(r_cox, 1),
#                 "rsf": clean_float(r_rsf, 1)
#             },
#             "curveData": [
#                 {"time": clean_float(t, 1), "cox": clean_float(c, 4), "rsf": clean_float(r, 4)} 
#                 for t, c, r in zip(grid, surv_cox_grid, surv_rsf_grid)
#             ]
#         }
#     except Exception as e:
#         print(f"Prediction Error: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/gene-intelligence")
# async def gene_intelligence():
#     """
#     Delegates to gene.py logic (v1 style)
#     """
#     try:
#         data = get_gene_intelligence(art)
#         return clean_nan_values(data)
#     except Exception as e:
#         print(f"Gene Intelligence Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/hero-graphs")
# async def hero_graphs_endpoint():
#     """
#     Delegates to graph.py logic (v1 style).
#     This works now because model_logic.py correctly creates 'group_cox'.
#     """
#     try:
#         data = get_hero_graphs(art)
#         return clean_nan_values(data)
#     except Exception as e:
#         print(f"Hero Graphs Error: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# # Optional: Feature Intelligence (Preserved from v2 if frontend needs it)
# @app.get("/feature-intelligence")
# async def feature_intelligence():
#     try:
#         df_cox = art["df_imp_cox"].head(30)
#         cox_data = [
#             {"feature": r["feature"], "value": clean_float(r["coef"], 4), "absVal": clean_float(r["abs_coef"], 4)}
#             for _, r in df_cox.iterrows()
#         ]
#         # RSF importance isn't strictly in model_logic return keys in your snippet,
#         # so we default to empty if missing to prevent crash.
#         rsf_data = [] 
        
#         return {"cox": cox_data, "rsf": rsf_data}
#     except Exception as e:
#         print(f"Feature Intelligence Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
