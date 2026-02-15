import os
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import traceback

# Import pure logic
from model_logic import (
    get_artifacts,
    median_survival_time,
    rmst,
    agreement_score,
    agreement_label
)
from graph import get_hero_graphs
from gene import get_gene_intelligence

app = FastAPI(title="OncoRisk Dual-Model API")

# --- CORS SECURITY ---
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip('/')

origins = [
    frontend_url,
    "https://onco-survival-ml-front-end.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALIZATION ---
print("Initializing Model Artifacts...")
try:
    art = get_artifacts()
    scaler = art["scaler"]
    cph = art["cph"]
    rsf = art["rsf"]
    features = art["features"]
    gene_features = art["gene_features"]
    print("Artifacts Loaded Successfully.")
except Exception as e:
    print(f"CRITICAL ERROR LOADING ARTIFACTS: {e}")
    traceback.print_exc()

# --- DATA MODELS ---
class InferenceRequest(BaseModel):
    age: float
    nodeStatus: str 
    genes: Dict[str, float]

# --- HELPER: THE FIX FOR JSON ERRORS ---
def clean_float(value):
    """
    Checks if a float is valid for JSON. 
    If it is NaN or Infinity, returns None (null in JSON).
    """
    if value is None:
        return None
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None

def clean_nan_values(obj):
    """
    Recursively cleans a complex object (dict/list) to replace NaNs with None.
    Critical for the /hero-graphs endpoint.
    """
    if isinstance(obj, float):
        return clean_float(obj)
    elif isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    elif isinstance(obj, (np.float64, np.float32)):
        return clean_float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    return obj

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "running", "allowed_origin": frontend_url}

@app.get("/metadata")
async def get_metadata():
    try:
        df_tcga = art["df_tcga"]
        test_ids = art["df_test"].index.tolist()[:100] 
        
        patients = []
        for pid in test_ids:
            if pid in df_tcga.index:
                patients.append({
                    "id": pid,
                    "age": float(df_tcga.loc[pid, "AGE"]),
                    "node": 1 if df_tcga.loc[pid, "NODE_POS"] == 1 else 0
                })
        
        return {"genes": gene_features, "patients": patients}
    except Exception as e:
        print(f"METADATA ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(data: InferenceRequest):
    print("\n--- NEW PREDICTION REQUEST ---")
    print(f"Input: Age={data.age}, Node={data.nodeStatus}")
    
    try:
        # 1. Feature Preparation
        node_pos = 1 if data.nodeStatus == "Positive" else 0
        row_dict = {"AGE": data.age, "NODE_POS": node_pos}
        row_dict.update(data.genes)
        
        row_raw = pd.DataFrame([row_dict])
        
        for feature in features:
            if feature not in row_raw.columns:
                row_raw[feature] = 0.0
        
        # Scale
        row_raw = row_raw[features]
        X_row = scaler.transform(row_raw.values)
        df_row_scaled = pd.DataFrame(X_row, columns=features)

        # 2. Cox Inference
        print("Running Cox Inference...")
        hazard_cox = float(cph.predict_partial_hazard(df_row_scaled).values[0])
        surv_funcs_cox = cph.predict_survival_function(df_row_scaled)
        times_cox = surv_funcs_cox.index.values
        surv_cox = surv_funcs_cox.values.flatten()
        median_cox = median_survival_time(times_cox, surv_cox)

        # 3. RSF Inference
        print("Running RSF Inference...")
        surv_funcs_rsf = rsf.predict_survival_function(X_row, return_array=True)
        times_rsf = rsf.unique_times_
        surv_rsf = surv_funcs_rsf[0]
        median_rsf = median_survival_time(times_rsf, surv_rsf)

        # 4. Metrics
        consensus_median = np.nanmean([median_cox, median_rsf])
        agree = agreement_score(median_cox, median_rsf)
        
        risk_mb_cox = cph.predict_partial_hazard(art["df_mb_s"]).values.flatten()
        percentile = (risk_mb_cox < hazard_cox).mean() * 100

        # 5. Graphs
        indices = np.linspace(0, len(times_cox) - 1, 50, dtype=int)
        rsf_interp = np.interp(times_cox[indices], times_rsf, surv_rsf)

        # --- DEBUG LOGGING STARTS HERE ---
        print("\n--- DEBUGGING CALCULATED VALUES ---")
        print(f"Cox Median (Raw): {median_cox}")
        print(f"RSF Median (Raw): {median_rsf}")
        print(f"Consensus (Raw): {consensus_median}")
        print(f"Agreement (Raw): {agree}")
        print(f"Percentile (Raw): {percentile}")
        
        # Check for NaNs in arrays
        if np.isnan(surv_cox[indices]).any():
            print("WARNING: NaN found in Cox Graph Data")
        if np.isnan(rsf_interp).any():
            print("WARNING: NaN found in RSF Graph Data")
        
        # --- SAFE RETURN WITH CLEANING ---
        # This converts any 'nan' to None (null) so JSON doesn't crash
        response = {
            "cox_median": clean_float(median_cox),
            "rsf_median": clean_float(median_rsf),
            "consensus_median": clean_float(consensus_median),
            "agreement_score": clean_float(agree),
            "agreement_label": agreement_label(agree),
            "risk_percentile": clean_float(percentile),
            "graph_data": {
                "times": [clean_float(t) for t in times_cox[indices]],
                "cox": [clean_float(c) for c in surv_cox[indices]],
                "rsf": [clean_float(r) for r in rsf_interp]
            }
        }
        
        print("Response prepared successfully.")
        return response

    except Exception as e:
        print(f"PREDICTION CRASHED: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

@app.get("/hero-graphs")
async def get_landing_graphs():
    try:
        from graph import get_hero_graphs
        data = get_hero_graphs(art)
        return data
    except Exception as e:
        print(f"HERO GRAPH ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
