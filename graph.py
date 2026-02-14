import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

def get_hero_graphs(art):
    """
    Generates optimized JSON data for the Landing Page Hero charts.
    1. TCGA KM Curves (Interpolated to common grid)
    2. METABRIC KM Curves (Interpolated to common grid)
    3. Cox vs RSF Scatter (Sampled for performance)
    """
    km = KaplanMeierFitter()

    def get_interpolated_km(df):
        # Create a common time grid (0 to max time, 60 steps) for smooth rendering
        t_max = df["time"].max()
        timeline = np.linspace(0, t_max, 60)
        
        # Fit Low Risk Group
        sub_low = df[df["group_cox"] == "Low"]
        km.fit(sub_low["time"], sub_low["event"])
        surv_low = km.survival_function_at_times(timeline).values
        
        # Fit High Risk Group
        sub_high = df[df["group_cox"] == "High"]
        km.fit(sub_high["time"], sub_high["event"])
        surv_high = km.survival_function_at_times(timeline).values
        
        # Structure for Recharts: [{time: 0, Low: 1.0, High: 1.0}, ...]
        data = []
        for t, s_l, s_h in zip(timeline, surv_low, surv_high):
            data.append({
                "time": int(t),
                "Low": round(float(s_l), 3),
                "High": round(float(s_h), 3)
            })
        return data

    # 1. TCGA Data
    tcga_data = get_interpolated_km(art["df_tcga_test_km"])

    # 2. METABRIC Data
    mb_data = get_interpolated_km(art["df_mb_km"])

    # 3. Scatter Data (Cox vs RSF)
    risk_cox = art["risk_test_cox"]
    risk_rsf = art["risk_test_rsf"]
    
    # Sample 300 points to keep the landing page animation fluid
    # (Using random choice to preserve density distribution)
    n_points = min(len(risk_cox), 300)
    indices = np.random.choice(len(risk_cox), n_points, replace=False)
    
    scatter_data = []
    for i in indices:
        scatter_data.append({
            "x": round(float(risk_cox[i]), 3),
            "y": round(float(risk_rsf[i]), 3)
        })

    return {
        "tcga_km": tcga_data,
        "metabric_km": mb_data,
        "scatter": scatter_data
    }