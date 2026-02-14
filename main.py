import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

# Import pure logic instead of app.py to avoid Streamlit errors
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

# --- CORS SECURITY CONFIGURATION ---
# In development, this allows localhost. In production (Render), it expects FRONTEND_URL.
# Default to localhost if variable is missing (safe for dev, override in Render).
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

origins = [
    frontend_url,
    "https://onco-survival-ml-front-end.vercel.app/clinical-implementation",
    "https://onco-survival-ml-front-end.vercel.app/riskprofile",
    "http://localhost:5173",  # Vite Local
    "http://localhost:3000",  # React Local
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # RESTRICTS access to these domains only
    allow_credentials=True,
    allow_methods=["*"],    # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],
)

# Initialize artifacts globally (Loaded once on startup)
print("Initializing Model Artifacts...")
art = get_artifacts()
scaler = art["scaler"]
cph = art["cph"]
rsf = art["rsf"]
features = art["features"]
gene_features = art["gene_features"]
print("Artifacts Loaded Successfully.")

# --- DATA MODELS ---
class InferenceRequest(BaseModel):
    age: float
    nodeStatus: str  # "Positive" or "Negative"
    genes: Dict[str, float]

@app.get("/")
def health_check():
    return {"status": "running", "allowed_origin": frontend_url}

@app.get("/metadata")
async def get_metadata():
    """Syncs the Frontend dropdowns with actual TCGA test patient IDs and genes."""
    df_tcga = art["df_tcga"]
    # Convert index to list and take first 50 for demo dropdown to keep payload light
    test_ids = art["df_test"].index.tolist()[:100] 
    
    patients = []
    for pid in test_ids:
        # Safety check if ID exists
        if pid in df_tcga.index:
            patients.append({
                "id": pid,
                "age": float(df_tcga.loc[pid, "AGE"]),
                "node": 1 if df_tcga.loc[pid, "NODE_POS"] == 1 else 0
            })
    
    return {
        "genes": gene_features,
        "patients": patients
    }

@app.post("/predict")
async def predict(data: InferenceRequest):
    try:
        # 1. Feature Preparation
        node_pos = 1 if data.nodeStatus == "Positive" else 0
        row_dict = {"AGE": data.age, "NODE_POS": node_pos}
        
        # Merge gene data
        row_dict.update(data.genes)
        
        # Create DataFrame and ensure column order matches training
        row_raw = pd.DataFrame([row_dict])
        
        # Check for missing columns and fill with 0 if necessary (safety)
        for feature in features:
            if feature not in row_raw.columns:
                row_raw[feature] = 0.0
                
        row_raw = row_raw[features]
        
        # Scale Data
        X_row = scaler.transform(row_raw.values)
        df_row_scaled = pd.DataFrame(X_row, columns=features)

        # 2. Cox Inference
        hazard_cox = float(cph.predict_partial_hazard(df_row_scaled).values[0])
        surv_funcs_cox = cph.predict_survival_function(df_row_scaled)
        times_cox = surv_funcs_cox.index.values
        surv_cox = surv_funcs_cox.values.flatten()
        median_cox = median_survival_time(times_cox, surv_cox)

        # 3. RSF Inference
        surv_funcs_rsf = rsf.predict_survival_function(X_row, return_array=True)
        times_rsf = rsf.unique_times_
        surv_rsf = surv_funcs_rsf[0]
        median_rsf = median_survival_time(times_rsf, surv_rsf)

        # 4. Consensus Metrics
        consensus_median = np.nanmean([median_cox, median_rsf])
        agree = agreement_score(median_cox, median_rsf)
        
        # 5. External Validation (Metabric) Check
        # Predicting on the whole MB cohort for context (optimized)
        # In a real high-load API, you might pre-calculate this
        risk_mb_cox = cph.predict_partial_hazard(art["df_mb_s"]).values.flatten()
        
        # Calculate percentiles
        user_risk_score = hazard_cox
        percentile = (risk_mb_cox < user_risk_score).mean() * 100

        # 6. Graphs Data (Simplified for JSON response)
        # Downsample for frontend performance
        indices = np.linspace(0, len(times_cox) - 1, 50, dtype=int)
        
        graph_data = {
            "times": times_cox[indices].tolist(),
            "cox": surv_cox[indices].tolist(),
            "rsf": np.interp(times_cox[indices], times_rsf, surv_rsf).tolist()
        }

        return {
            "cox_median": float(median_cox) if not np.isnan(median_cox) else None,
            "rsf_median": float(median_rsf) if not np.isnan(median_rsf) else None,
            "consensus_median": float(consensus_median) if not np.isnan(consensus_median) else None,
            "agreement_score": float(agree),
            "agreement_label": agreement_label(agree),
            "risk_percentile": float(percentile),
            "graph_data": graph_data
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
