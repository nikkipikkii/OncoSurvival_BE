import os
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import traceback

# --- IMPORT LOGIC FROM YOUR MODULES ---
from graph import get_hero_graphs
from gene import get_gene_intelligence
from model_logic import (
    get_artifacts,
    median_survival_time,
    rmst,
    agreement_score,
    agreement_label
)

app = FastAPI(title="OncoRisk Dual-Model API")

# --- CORS SECURITY ---
# In deployment, this allows the frontend to talk to the backend
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip('/')
origins = [
    frontend_url,
    "https://onco-survival-ml-front-end.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Wildcard is safer for debugging; restrict in strict prod
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
    features = art["features"]       # The list of feature names in correct order
    gene_features = art["gene_features"] # The list of just the 31 genes
    print("Artifacts Loaded Successfully.")
except Exception as e:
    print(f"CRITICAL ERROR LOADING ARTIFACTS: {e}")
    traceback.print_exc()

# --- DATA MODELS ---
class InferenceRequest(BaseModel):
    age: float
    nodeStatus: str 
    genes: Dict[str, float]

# --- TYPE-SYNC SAFETY HELPERS (PREVENTS CRASHES) ---
def clean_float(value):
    """
    Checks if a float is valid for JSON (handles NaN/Inf/Numpy types).
    Returns None if invalid, which JSON accepts as 'null'.
    """
    if value is None:
        return None
    try:
        val = float(value) # Force conversion from np.float to python float
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None

def clean_nan_values(obj):
    """
    Recursively cleans a complex object (dict/list) to replace NaNs with None.
    Crucial for nested responses like hero-graphs or gene-intelligence.
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
    """
    Returns patient list and gene list. 
    Includes 'genes' in patient object so Auto-Fill works in frontend.
    """
    try:
        df_tcga = art["df_tcga"]
        test_ids = art["df_test"].index.tolist()[:100] 
        
        patients = []
        for pid in test_ids:
            if pid in df_tcga.index:
                # Extract actual gene values for this patient to enable auto-fill
                patient_genes = df_tcga.loc[pid, gene_features].to_dict()
                
                patients.append({
                    "id": pid,
                    "age": clean_float(df_tcga.loc[pid, "AGE"]),
                    "node": 1 if df_tcga.loc[pid, "NODE_POS"] == 1 else 0,
                    "genes": clean_nan_values(patient_genes) 
                })
        
        return {"genes": gene_features, "patients": patients}
    except Exception as e:
        print(f"METADATA ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(data: InferenceRequest):
    """
    The Core Inference Logic. 
    1. Feature Engineering (Age + Node + Genes)
    2. Scaling (StandardScaler)
    3. Model Prediction (Cox + RSF)
    4. Grid Synchronization (200 points)
    5. Metric Calculation (Median, RMST, Agreement)
    6. Response Formatting (Strict TypeSync)
    """
    try:
        # --- 1. FEATURE PREPARATION ---
        node_pos = 1 if data.nodeStatus == "Positive" else 0
        row_dict = {"AGE": data.age, "NODE_POS": node_pos}
        row_dict.update(data.genes)
        
        # Create DataFrame ensuring EXACT column order required by Scaler
        row_raw = pd.DataFrame([row_dict], columns=features)
        
        # Safeguard: Fill missing columns with 0.0 to prevent crash
        row_raw = row_raw.fillna(0.0)
        
        # Scale
        X_row = scaler.transform(row_raw.values)
        df_row_scaled = pd.DataFrame(X_row, columns=features)

        # --- 2. MODEL INFERENCE ---
        
        # Cox Hazard
        hazard_cox = float(cph.predict_partial_hazard(df_row_scaled).values[0])
        surv_func_cox = cph.predict_survival_function(df_row_scaled)
        # Extract times and probabilities (Step function)
        times_cox = surv_func_cox.index.values.astype(float)
        surv_cox = surv_func_cox.values[:, 0].astype(float)

        # RSF Risk
        surv_funcs_rsf = rsf.predict_survival_function(X_row)
        sf_rsf = surv_funcs_rsf[0]
        # Extract times and probabilities (Step function)
        times_rsf = sf_rsf.x.astype(float)
        surv_rsf = sf_rsf.y.astype(float)

        # --- 3. GRID SYNCHRONIZATION (The "Original Logic") ---
        # We must interpolate both models onto a shared time grid to calculate 
        # consensus and draw the graph correctly.
        t_min = 0.0
        t_max = min(times_cox.max(), times_rsf.max())
        
        # Create a common grid of 200 points
        grid = np.linspace(t_min, t_max, 200) 

        # Interpolate
        surv_cox_grid = np.interp(grid, times_cox, surv_cox)
        surv_rsf_grid = np.interp(grid, times_rsf, surv_rsf)

        # --- 4. METRIC EXTRACTION ---
        m_cox = median_survival_time(grid, surv_cox_grid)
        m_rsf = median_survival_time(grid, surv_rsf_grid)
        
        r_cox = rmst(grid, surv_cox_grid)
        r_rsf = rmst(grid, surv_rsf_grid)
        
        # Consensus & Agreement
        consensus_median = np.nanmean([m_cox, m_rsf])
        agree = agreement_score(m_cox, m_rsf)

        # Logging for Debugging
        print(f"Inference -> Age: {data.age}, Node: {data.nodeStatus}")
        print(f"Cox Median: {m_cox}, RSF Median: {m_rsf}, Agreement: {agree}")

        # --- 5. RESPONSE CONSTRUCTION (STRICT TYPESYNC) ---
        # We manually build the dict using `clean_float` and `float()` casting 
        # to ensure no Numpy types leak into the JSON response.
        
        response = {
            "summary": {
                "coxHazard": clean_float(hazard_cox),
                   "rsfRisk": round(float(sf_rsf.y[-1]), 2),
                # "rsfRisk": clean_float(1.0 - float(surv_rsf_grid[-1])), # Risk = 1 - Survival
                "agreement": clean_float(agree),
                "agreementLabel": agreement_label(agree)
            },
            "estimates": {
                # "medianCox": clean_float(m_cox),
                # "medianRsf": clean_float(m_rsf),
                # "consensus": clean_float(consensus_median)
                "medianCox": clean_float(m_cox, 1),
                "medianRsf": clean_float(m_rsf, 1),
                "consensus": clean_float(consensus_median, 1)
            },
            "rmst": {
                # "cox": clean_float(r_cox),
                # "rsf": clean_float(r_rsf)
                "cox": clean_float(r_cox, 1),
                "rsf": clean_float(r_rsf, 1)
            },
            # Graph Data formatted specifically for Recharts
            "curveData": [
                {
                    "time": float(t), 
                    "cox": float(c), 
                    "rsf": float(r)
                } 
                for t, c, r in zip(grid, surv_cox_grid, surv_rsf_grid)
            ]
        }
        
        return response

    except Exception as e:
        print(f"PREDICTION ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gene-intelligence")
async def get_gene_data():
    try:
        if art is None:
             raise HTTPException(status_code=500, detail="Model artifacts not loaded")
        
        data = get_gene_intelligence(art)
        # SAFEGUARD: Recursively clean NaNs
        return clean_nan_values(data)
    except Exception as e:
        print(f"GENE INTELLIGENCE ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hero-graphs")
async def get_landing_graphs():
    try:
        data = get_hero_graphs(art)
        # SAFEGUARD: Recursively clean NaNs
        return clean_nan_values(data)
    except Exception as e:
        print(f"HERO GRAPH ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Local development runner
    uvicorn.run(app, host="0.0.0.0", port=8000)
# import os
# import math
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, List, Optional
# import traceback
# from graph import get_hero_graphs
# from gene import get_gene_intelligence

# # Import pure logic
# #import from app-> get_arti, median_surviva_rmst,agre_score,agr_label 
# from model_logic import (
#     get_artifacts,
#     median_survival_time,
#     rmst,
#     agreement_score,
#     agreement_label
# )
# # from graph import get_hero_graphs
# # from gene import get_gene_intelligence

# app = FastAPI(title="OncoRisk Dual-Model API")

# # --- CORS SECURITY ---
# frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip('/')
# # --------------------extra
# origins = [
#     frontend_url,
#     "https://onco-survival-ml-front-end.vercel.app",
#     "http://localhost:5173",
#     "http://localhost:3000",
# ]
# # --------extra^
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # Keep wildcard for debugging, restrict in production if needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- INITIALIZATION ---
# print("Initializing Model Artifacts...")
# try:
#     art = get_artifacts()
#     scaler = art["scaler"]
#     cph = art["cph"]
#     print(cph);
#     rsf = art["rsf"]
#     features = art["features"]
#     gene_features = art["gene_features"]
#     print("Artifacts Loaded Successfully.")
# except Exception as e:
#     print(f"CRITICAL ERROR LOADING ARTIFACTS: {e}")
#     traceback.print_exc()

# # --- DATA MODELS ---
# class InferenceRequest(BaseModel):
#     age: float
#     nodeStatus: str 
#     genes: Dict[str, float]

# # --- HELPER: THE FIX FOR JSON ERRORS ---
# def clean_float(value):
#     """
#     Checks if a float is valid for JSON. 
#     If it is NaN or Infinity, returns None (null in JSON).
#     """
#     if value is None:
#         return None
#     try:
#         val = float(value)
#         if math.isnan(val) or math.isinf(val):
#             return None
#         return val
#     except Exception:
#         return None

# def clean_nan_values(obj):
#     """
#     Recursively cleans a complex object (dict/list) to replace NaNs with None.
#     Critical for the /hero-graphs endpoint.
#     """
#     if isinstance(obj, float):
#         return clean_float(obj)
#     elif isinstance(obj, dict):
#         return {k: clean_nan_values(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [clean_nan_values(v) for v in obj]
#     elif isinstance(obj, (np.float64, np.float32)):
#         return clean_float(obj)
#     elif isinstance(obj, (np.int64, np.int32)):
#         return int(obj)
#     return obj

# # --- ENDPOINTS ---

# @app.get("/")
# def health_check():
#     return {"status": "running", "allowed_origin": frontend_url}

# @app.get("/metadata")
# async def get_metadata():
#     try:
#         df_tcga = art["df_tcga"]
#         test_ids = art["df_test"].index.tolist()[:100] 
        
#         patients = []
#         for pid in test_ids:
#             if pid in df_tcga.index:
#                 patients.append({
#                     "id": pid,
#                     "age": float(df_tcga.loc[pid, "AGE"]),
#                     "node": 1 if df_tcga.loc[pid, "NODE_POS"] == 1 else 0
#                 })
        
#         return {"genes": gene_features, "patients": patients}
#     except Exception as e:
#         print(f"METADATA ERROR: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/predict")
# async def predict(data: InferenceRequest):
#     print("\n--- NEW PREDICTION REQUEST ---")
#     print(f"Input: Age={data.age}, Node={data.nodeStatus}")
    
#     try:
#         # 1. Feature Preparation
#         node_pos = 1 if data.nodeStatus == "Positive" else 0
#         row_dict = {"AGE": data.age, "NODE_POS": node_pos}
#         row_dict.update(data.genes)
        
#         row_raw = pd.DataFrame([row_dict])
#         #  row_raw = pd.DataFrame([row_dict])[features]
#         for feature in features:
#             if feature not in row_raw.columns:
#                 row_raw[feature] = 0.0
#         # the actual logic is
#         #         row_raw = pd.DataFrame([row_dict])[features]
#         # X_row = scaler.transform(row_raw.values)
#         # df_row_scaled = pd.DataFrame(X_row, columns=features)

#         # Scale
#         row_raw = row_raw[features]
#         X_row = scaler.transform(row_raw.values)
#         df_row_scaled = pd.DataFrame(X_row, columns=features)

#         # 2. Cox Inference
#         # it is Cox Hazard
#         print("Running Cox Inference...")
#         hazard_cox = float(cph.predict_partial_hazard(df_row_scaled).values[0])
#         surv_funcs_cox = cph.predict_survival_function(df_row_scaled)
#         times_cox = surv_funcs_cox.index.values
#         # times_cox = surv_func_cox.index.values.astype(float)
#         surv_cox = surv_funcs_cox.values.flatten()
#         # why has the values been faltted?
#         #    surv_cox = surv_func_cox.values[:, 0].astype(float)
#         median_cox = median_survival_time(times_cox, surv_cox)
#         # median cox us not under cox indrence

#         # 3. RSF Inference
#         print("Running RSF Inference...")
#         surv_funcs_rsf = rsf.predict_survival_function(X_row, return_array=True)
#         surv_rsf = surv_funcs_rsf[0]
#         times_rsf = rsf.unique_times_
#         median_rsf = median_survival_time(times_rsf, surv_rsf)
#         # mediam rsf not in logic nstead
#                # RSF Risk
#         # surv_funcs_rsf = rsf.predict_survival_function(X_row)
#         # sf_rsf = surv_funcs_rsf[0]
#         # times_rsf = sf_rsf.x.astype(float)
#         # surv_rsf = sf_rsf.y.astype(float)

#         # 4. Metrics
#         consensus_median = np.nanmean([median_cox, median_rsf])
#         agree = agreement_score(median_cox, median_rsf)
        
#         risk_mb_cox = cph.predict_partial_hazard(art["df_mb_s"]).values.flatten()
#         percentile = (risk_mb_cox < hazard_cox).mean() * 100

#         # 5. Graphs
#         indices = np.linspace(0, len(times_cox) - 1, 50, dtype=int)
#         rsf_interp = np.interp(times_cox[indices], times_rsf, surv_rsf)
        
#         print("\n--- DEBUGGING CALCULATED VALUES ---")

#         print(f"Cox Median (Raw): {median_cox}")

#         print(f"RSF Median (Raw): {median_rsf}")

#         print(f"Consensus (Raw): {consensus_median}")

#         print(f"Agreement (Raw): {agree}")

#         print(f"Percentile (Raw): {percentile}")

        

#         # Check for NaNs in arrays

#         if np.isnan(surv_cox[indices]).any():

#             print("WARNING: NaN found in Cox Graph Data")

#         if np.isnan(rsf_interp).any():

#             print("WARNING: NaN found in RSF Graph Data")
#         # --- SAFE RETURN WITH CLEANING ---
#         response = {
#             "cox_median": clean_float(median_cox),
#             "rsf_median": clean_float(median_rsf),
#             "consensus_median": clean_float(consensus_median),
#             "agreement_score": clean_float(agree),
#             "agreement_label": agreement_label(agree),
#             "risk_percentile": clean_float(percentile),
#             "graph_data": {
#                 "times": [clean_float(t) for t in times_cox[indices]],
#                 "cox": [clean_float(c) for c in surv_cox[indices]],
#                 "rsf": [clean_float(r) for r in rsf_interp]
#             }
#         }
        
#         print("Response prepared successfully.")
#         return response

#     except Exception as e:
#         print(f"PREDICTION CRASHED: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


# # ====ACTUAL LOGIC BEFORE DEPLOYMENT

# # =================================================================
# # NEW ENDPOINT: GENE INTELLIGENCE
# # =================================================================
# @app.get("/gene-intelligence")
# async def get_gene_data():
#     try:
#         # Check if art exists
#         if art is None:
#              raise HTTPException(status_code=500, detail="Model artifacts not loaded")
        
#         data = get_gene_intelligence(art)
#         # CRITICAL FIX: Clean NaNs before sending JSON
#         return clean_nan_values(data)
#     except Exception as e:
#         print(f"GENE INTELLIGENCE ERROR: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# # =================================================================
# # FIXED ENDPOINT: HERO GRAPHS (Added clean_nan_values)
# # =================================================================
# @app.get("/hero-graphs")
# async def get_landing_graphs():
#     try:
#         from graph import get_hero_graphs
#         data = get_hero_graphs(art)
#         # CRITICAL FIX: Clean NaNs to prevent 500 Internal Server Error
#         return clean_nan_values(data)
#     except Exception as e:
#         print(f"HERO GRAPH ERROR: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))
