import streamlit as st

st.set_page_config(
    page_title="OncoRisk™",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.success("✅ THIS app.py is running")



# ============================================================
# OncoRisk – Dual Cox + RSF Clinical Survival Platform
# ============================================================

import os
from pathlib import Path
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def load_bg_image(path):
    path = Path(path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored


# ============================================================
# SESSION STATE (SINGLE SOURCE OF TRUTH)
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Overview"

PAGES = [
    "Overview",
    "Dashboard",
    "Patient Risk Profile Inference",
    "Cohort Analytics",
    "Feature Intelligence",
    "Gene Intelligence",
    "Model Facts",
]

# ============================================================
# URL → SESSION STATE SYNC (REQUIRED FOR CTA NAVIGATION)
# ============================================================
# ---- URL → session sync (CTA support) ----
if "page" in st.query_params:
    qp = st.query_params["page"]
    if qp in PAGES:
        st.session_state.page = qp


st.sidebar.title("OncoRisk Dual")
st.sidebar.markdown("Cox + RSF survival intelligence")
        
page = st.sidebar.selectbox(
    "Navigation",
    PAGES,
    index=PAGES.index(st.session_state.page),
)

st.session_state.page = page

st.write("🔎 PAGE VALUE =", repr(page))




# ============================================================
# GLOBAL STYLE
# ============================================================

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

st.markdown("""
<style>

/* ===== STREAMLIT HEADERS (st.header / st.subheader) ===== */
h1[data-testid="stHeader"] {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    margin-top: 3.5rem !important;
    margin-bottom: 1.5rem !important;
    color: #ffffff !important;
}

h2[data-testid="stHeader"] {
    font-size: 2.1rem !important;
    font-weight: 750 !important;
    margin-top: 3.0rem !important;
    margin-bottom: 1.3rem !important;
    color: #ffffff !important;
}

h3[data-testid="stHeader"] {
    font-size: 1.6rem !important;
    font-weight: 650 !important;
    margin-top: 2.2rem !important;
    margin-bottom: 1.0rem !important;
    color: #ffffff !important;
}
/* ===== HERO CTA BUTTON ===== */
.hero-cta {
    display: inline-block;
    margin-top: 2rem;
    padding: 0.9rem 1.6rem;
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    background: linear-gradient(
        135deg,
        #e5e7eb 0%,
        #ffffff 100%
    );
    border-radius: 999px;
    text-decoration: none;
    box-shadow: 
        0 10px 25px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.6);
    transition: all 0.25s ease;
}

.hero-cta:hover {
    transform: translateY(-2px);
    box-shadow:
        0 14px 30px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.7);
}

.hero-cta:active {
    transform: translateY(0);
}


/* Kill Streamlit spacing suppression */

/* ===== EXPANDER HEADERS ===== */
button[aria-expanded] {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* ===== BODY TEXT ===== */
/* ===== MAIN APP BACKGROUND & TEXT ===== */
[data-testid="stAppViewContainer"] {
    background-color: #111418;
    color: #e5e7eb;
}

/* Main content text */
.block-container {
    color: #e5e7eb;
}


/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #151a20;
    border-right: 1px solid #2b2f38;
}

/* Layout */
.block-container {
    padding: 1.5rem 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Force readable text everywhere */
body, p, div, span, li {
    color: #e5e7eb !important;
}

/* Titles */
h1, h2, h3, h4 {
    color: #f8fafc !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}
[data-testid="stMetricLabel"] {
    color: #cbd5f5 !important;
}

/* Captions */
[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "Models"
SPLITS_DIR = MODELS_DIR / "CoxPH_Final" / "splits"

CLINICOGENOMIC_DIR = MODELS_DIR / "Clinicogenomic_31genes_v2"
TCGA_PATH = CLINICOGENOMIC_DIR / "tables" / "tcga_clinicogenomic_31genes_with_surv.csv"
MB_PATH   = CLINICOGENOMIC_DIR / "tables" / "metabric_clinicogenomic_31genes_with_surv.csv"

# ============================================================
# SURVIVAL HELPERS
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

# ============================================================
# DATA + MODEL LOADING
# ============================================================

@st.cache_resource
def load_artifacts():
    df_tcga = pd.read_csv(TCGA_PATH, index_col=0)
    df_mb   = pd.read_csv(MB_PATH, index_col=0)

    clinical = ["AGE", "NODE_POS"]
    
    gene_features = [
        c for c in df_tcga.columns
        if c not in ["time", "event", "AGE", "NODE_POS"]
    ]
    features = clinical + gene_features

    train_ids = pd.read_csv(SPLITS_DIR / "train_ids.csv").iloc[:, 0]
    test_ids  = pd.read_csv(SPLITS_DIR / "test_ids.csv").iloc[:, 0]

    train_ids = train_ids[train_ids.isin(df_tcga.index)]
    test_ids  = test_ids[test_ids.isin(df_tcga.index)]

    df_train = df_tcga.loc[train_ids]
    df_test  = df_tcga.loc[test_ids]

    scaler = StandardScaler().fit(df_train[features])

    X_train = scaler.transform(df_train[features])
    X_test  = scaler.transform(df_test[features])
    X_mb    = scaler.transform(df_mb[features])

    df_train_s = pd.concat(
        [df_train[["time","event"]],
         pd.DataFrame(X_train, index=df_train.index, columns=features)], axis=1)

    df_test_s = pd.concat(
        [df_test[["time","event"]],
         pd.DataFrame(X_test, index=df_test.index, columns=features)], axis=1)

    df_mb_s = pd.concat(
        [df_mb[["time","event"]],
         pd.DataFrame(X_mb, index=df_mb.index, columns=features)], axis=1)

    # Cox
    cph = CoxPHFitter()
    cph.fit(df_train_s, "time", "event")

    risk_train_cox = cph.predict_partial_hazard(df_train_s).values.flatten()
    risk_test_cox  = cph.predict_partial_hazard(df_test_s).values.flatten()
    risk_mb_cox    = cph.predict_partial_hazard(df_mb_s).values.flatten()
    cindex_train_cox = concordance_index(
        df_train_s["time"],
        -risk_train_cox,
        df_train_s["event"]
    )


    cindex_test_cox  = concordance_index(df_test_s["time"],  -risk_test_cox,  df_test_s["event"])
    cindex_mb_cox    = concordance_index(df_mb_s["time"],    -risk_mb_cox,    df_mb_s["event"])

    # RSF
    rsf = RandomSurvivalForest(
        n_estimators=500,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=123,
        n_jobs=-1
    )
    rsf.fit(X_train, Surv.from_arrays(df_train["event"].astype(bool),
                                      df_train["time"].astype(float)))

    ch_train = rsf.predict_cumulative_hazard_function(X_train)
    ch_test  = rsf.predict_cumulative_hazard_function(X_test)
    ch_mb    = rsf.predict_cumulative_hazard_function(X_mb)

    risk_train_rsf = np.array([fn.y[-1] for fn in ch_train])
    risk_test_rsf  = np.array([fn.y[-1] for fn in ch_test])
    risk_mb_rsf    = np.array([fn.y[-1] for fn in ch_mb])


    cindex_train_rsf = concordance_index_censored(
        df_train["event"].astype(bool),
        df_train["time"].astype(float),
        risk_train_rsf)[0]

    cindex_test_rsf = concordance_index_censored(
        df_test["event"].astype(bool),
        df_test["time"].astype(float),
        risk_test_rsf)[0]

    cindex_mb_rsf = concordance_index_censored(
        df_mb["event"].astype(bool),
        df_mb["time"].astype(float),
        risk_mb_rsf)[0]

    # Feature importance
    df_imp_cox = cph.params_.to_frame("coef")
    df_imp_cox["feature"] = df_imp_cox.index
    df_imp_cox["abs_coef"] = df_imp_cox["coef"].abs()
    df_imp_cox["type"] = ["clinical" if f in clinical else "gene" for f in df_imp_cox.index]
    df_imp_cox = df_imp_cox.sort_values("abs_coef", ascending=False)


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

    # KM prep
    df_tcga_test_km = df_test_s.copy()
    df_tcga_test_km["risk_cox"] = risk_test_cox
    cut_tcga = np.median(risk_test_cox)
    df_tcga_test_km["group_cox"] = np.where(df_tcga_test_km["risk_cox"] > cut_tcga, "High", "Low")

    df_mb_km = df_mb_s.copy()
    df_mb_km["risk_cox"] = risk_mb_cox
    cut_mb = np.median(risk_mb_cox)
    df_mb_km["group_cox"] = np.where(df_mb_km["risk_cox"] > cut_mb, "High", "Low")

    return dict(
        df_tcga=df_tcga,
        df_mb=df_mb,
        df_test=df_test_s,
        df_mb_s=df_mb_s,
        features=features,
        gene_features=gene_features,
        scaler=scaler,
        cph=cph,
        rsf=rsf,
        risk_test_cox=risk_test_cox,
        risk_test_rsf=risk_test_rsf,
        cindex_train_cox=cindex_train_cox,
        cindex_train_rsf=cindex_train_rsf,
        df_tcga_test_km=df_tcga_test_km,
        df_mb_km=df_mb_km,
        df_imp_cox=df_imp_cox,
        df_imp_rsf=df_imp_rsf,
        cindex_test_cox=cindex_test_cox,
        cindex_test_rsf=cindex_test_rsf,
        cindex_mb_cox=cindex_mb_cox,
        cindex_mb_rsf=cindex_mb_rsf
    )
@st.cache_resource(show_spinner="Loading survival models...")
def get_artifacts():
    return load_artifacts()


# ============================================================
# GENE NARRATIVES
# ============================================================

GENE_NARRATIVES = {
    "CD24": "Immune evasion via Siglec-10; higher expression aligns with increased hazard.",
    "SLC16A2": "Thyroid hormone transport; metabolic rewiring under stress.",
    "SERPINA1": "Protease regulation and invasion-permissive microenvironment.",
    "TFPI2": "Extracellular matrix control and metastatic potential.",
    "SEMA3B": "Loss of repulsive anti-invasive cues.",
    "TNFRSF14": "Tumor–immune interface regulation.",
    "APOOL": "Mitochondrial lipid handling.",
    "MRPL13": "Oxidative phosphorylation dependency.",
    "QPRT": "NAD+ biosynthesis under stress.",
    "JCHAIN": "Humoral immune architecture.",
    "NANOS1": "Stem-like, therapy-tolerant state.",
    "EDA2R": "Stress-adaptive signaling.",
}

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "Overview":
    st.write("🧪 ENTERED:", page)

    dna_bg = load_bg_image(BASE_DIR / "dna_hero.png")


    st.markdown(
        f"""
<div style="
position: relative;
height: 48vh;
max-height: 460px;
width: 100%;
border-radius: 20px;
overflow: hidden;
margin-bottom: 3rem;
background:
    linear-gradient(
        90deg,
        rgba(10,12,20,0.65) 20%,
        rgba(10,12,20,0.30) 65%
    ),
    url('data:image/png;base64,{dna_bg}') left center / cover no-repeat;
display: flex;
align-items: center;
">
<div style="
padding: 4rem;
max-width: 55%;
background: rgba(15,18,24,0.35);
backdrop-filter: blur(6px);
border-radius: 14px;
">
<h1 style="margin-bottom:0.5rem;text-shadow:0 2px 6px rgba(0,0,0,0.5);">
OncoRisk
</h1>

<h3 style="margin-top:0;text-shadow:0 1px 4px rgba(0,0,0,0.45);">
Clinicogenomic Survival Intelligence
</h3>

<p style="margin-top:1rem;color:#f1f5f9;max-width:36ch;">
Explore survival structure, individual risk profiles,
and biological drivers across independent cohorts.
</p>
</div>
</div>
""",
        unsafe_allow_html=True
    )


    st.markdown(
    """
    <a class="hero-cta" href="?page=Dashboard">
        → Explore the Model
    </a>
    """,
        unsafe_allow_html=True
    )



    st.markdown("---")

    # ---------- SECTION 1 ----------
    st.header("What This Project Is")
    st.markdown("""
OncoRisk is a research-grade survival modeling platform developed to study how a compact set of
clinical variables and tumor gene expression jointly shape long-term survival risk in breast cancer.
Rather than producing binary predictions or treatment recommendations, the system models
**time-to-event behavior** using established survival analysis frameworks. Its outputs include
relative hazard estimates, full survival curves, and survival-time summaries such as median survival
and restricted mean survival time (RMST).
The platform was developed using TCGA-BRCA and evaluated on an entirely independent external cohort,
METABRIC, spanning more than 2,900 patients in total. The goal is not maximal prediction within a
single dataset, but **interpretability, stability, and external validity across cohorts**.
""")
    st.markdown("---")

    # ---------- SECTION 2 ----------
    st.header("Why This Model Exists")
    st.caption("The motivating question behind the modeling choices")

    st.markdown("""
Clinical staging and basic pathology explain only part of why patients with similar diagnoses
experience markedly different outcomes. Molecular features can add information, but many published
signatures rely on hundreds or thousands of genes, making them difficult to interpret and fragile
outside their training cohort.
""")
    st.markdown("""
### ❓ Core Research Question

**Can a compact, biologically coherent clinicogenomic feature set support stable and explainable
survival modeling across independent breast cancer cohorts?**
""")

    st.markdown("""
- inputs were limited to variables that repeatedly demonstrated signal under stress-testing
- transparent survival models were favored over opaque predictors
- agreement across linear and nonlinear modeling paradigms was treated as informative
- external validation was prioritized over optimization on TCGA alone
""")
    st.markdown("---")

    # ---------- SECTION 3 ----------
    st.header("Modeling Strategy: A Dual-Model System")
    st.markdown("""
OncoRisk uses two complementary survival models, each serving a distinct role.
The primary risk engine is a Cox proportional hazards model, chosen for its long-standing role in clinical survival analysis, explicit coefficients, and stable behavior under external validation. From this model, the system derives relative hazard scores, full survival curves, median survival time, and restricted mean survival time.
Alongside Cox, a Random Survival Forest (RSF) is trained on the same feature set to capture nonlinear effects and feature interactions that linear models cannot represent. RSF produces independent survival curves and survival-time estimates but is not used as a replacement for Cox.
Instead, the two models form a dual lens: Cox provides a mechanistic and interpretable hazard structure, while RSF offers flexibility and interaction awareness. For each patient-like profile, the app shows both CoxPH and RSF survival curves, median survival, RMST, and a simple agreement score between the two is treated as informative, and disagreement is made explicit rather than hidden.
""")

    st.markdown("---")

    # ---------- SECTION 4 ----------
    st.header("Data and Cohorts")

    st.subheader("TCGA-BRCA (Training and Internal Evaluation)")
    st.markdown("""
- **1,059 patients**
- **Training set:** 847 patients (120 observed events)
- **Test set:** 212 patients (30 observed events)
""")

    st.subheader("METABRIC (External Validation)")
    st.markdown("""
- **1,903 patients**
- **1,103 observed events**
- No samples used during training or tuning
""")

    st.markdown("""
The final input space consisted of **33 features per patient**.

**Clinical (2):**
- Age at diagnosis
- Lymph-node involvement (binary)

**Molecular (31):**
- A curated gene expression panel present in both TCGA and METABRIC
  In total, the model operates on 33 features per patient.
  This design is intentional: fewer variables, clearer interpretation
""")
    st.markdown("---")


    # ============================================================
    # EXPANDER 1
    # ============================================================

    with st.expander("🔍 Model Design & Inference Logic"):
        st.subheader("Data and Cohorts")
        st.markdown("""Model development and evaluation used two large, independent breast cancer cohorts that differ in origin, measurement technology, and clinical context. This separation was intentional: one cohort was used for model development and internal evaluation, while the other was reserved exclusively for external validation.
**TCGA-BRCA (Used for Training and Internal Evaluation)**
The primary development cohort was TCGA-BRCA, a genomically profiled breast cancer dataset generated by The Cancer Genome Atlas. After aligning survival outcomes, clinical variables, and gene expression data, the final TCGA cohort included 1,059 patients with complete information.
To prevent information leakage and preserve reproducibility, a frozen train–test split was used throughout the project:
* Training set: 847 patients (120 observed events)
* Test set: 212 patients (30 observed events)
All model fitting, feature scaling, and parameter estimation were performed strictly on the training set. The TCGA test set was used only for internal evaluation and visualization, reflecting how the model would behave on unseen patients from the same source.
**METABRIC (Used exclusively for External Validation)**
METABRIC is an independently collected breast cancer cohort generated using microarray-based gene expression profiling and long-term clinical follow-up. No METABRIC samples were used during model training, feature selection, or hyperparameter tuning.
Its role was to test whether the survival structure learned from TCGA generalizes across differences in cohort composition, measurement technology, and clinical practice.
After harmonizing clinical variables and restricting analysis to genes shared with TCGA, the final METABRIC cohort included 1,903 patients, with 1,103 observed events. Gene expression was restricted to a curated 31-gene signature present in both cohorts, enforcing cross-cohort compatibility and reducing overfitting risk.""")

        st.subheader("Patient-Level Profile Inference")
        st.markdown("""For any clinicogenomic profile, the system computes survival curves and time-based summaries from both models, including median survival and RMST. A simple agreement score quantifies how closely the two models align.
This enables a simple twin-model logic: when both survival models trained on the same data tell a similar survival story, confidence increases; when they diverge, uncertainty is exposed rather than smoothed away. No single number is treated as absolute truth, which keeps the output aligned with research use rather than clinical decision-making.
""")
    st.markdown("---")

    # ============================================================
    # EXPANDER 2
    # ============================================================

    with st.expander("🧬 Biological Programs and Survival Structure"):
        st.subheader("Why These Inputs Were Chosen")
        st.markdown(""" **Lymph-Node Involvement**
Lymph-node status is one of the strongest real-world predictors of prognosis in breast cancer because it captures whether tumor cells have escaped the primary site and entered the lymphatic system. In this project’s experiments, including lymph-node positivity consistently improved separation between low- and high-risk groups and strengthened external performance, making it a central anchor rather than a secondary covariate.

**Age at Diagnosis**
Age captures a mixture of immune, hormonal, and treatment-related effects that shift baseline risk even when tumors appear similar histologically. Across TCGA and METABRIC, age showed a stable, directionally consistent association with hazard and remained reliable under external validation, justifying its inclusion as a simple but robust clinical modifier.

**The 31-Gene Panel**
Instead of using thousands of genes, OncoRisk relies on a curated 31-gene panel present in both cohorts. Across Cox proportional hazards models, Random Survival Forests, and gradient-boosted survival models, these genes kept repeatedly contributed non-random, directionally consistent signal.
Functionally, they cluster into three recurring biological programs:
* **Immune regulation and the tumor–immune interface**: genes tied to how tumors avoid or reshape immune attack.(e.g., CD24, TNFRSF14, CCL19)
* **Metabolic plasticity and stress tolerance**: genes that support energy rewiring and survival under stress.​ (e.g., SLC16A2, SERPINA1, QPRT)
* **Extracellular matrix remodeling and invasion**: genes involved in reshaping tissue structure to support spread and immune escape (e.g., SEMA3B, TFPI2)
Activation of these programs consistently aligned with worse survival, which is why this panel became the molecular backbone of the system.
""")

        st.subheader("What the Model Learns")
        st.markdown("""Across cohorts, both models consistently surface three dominant survival programs:

* immune regulation at the tumor–immune interface
* metabolic stress tolerance and mitochondrial function
* extracellular matrix remodeling associated with invasion

High-risk tumors tend to show coordinated activation of these programs, while low-risk tumors lack a dominant driver profile. This pattern suggests that aggressive disease follows structured biological strategies rather than arising from random molecular noise.""")
    st.markdown("---")

    # ============================================================
    # EXPANDER 3
    # ============================================================

    with st.expander("📊 Outputs, Validation, and Scope"):
        st.subheader("Outputs and Intended Use")
        st.markdown("""The system reports:
* relative hazard estimates
* nonlinear risk measures
* survival curves over time
* median survival time
* restricted mean survival time (RMST)
* a model-agreement score
It deliberately does not output treatment recommendations, diagnostic classifications, or absolute survival probabilities intended for decision-making. All outputs are designed for research, stratification, and hypothesis generation.
""")

        st.subheader("What This System Is — and Is Not")
        st.markdown("""**This system is:**
* a dual-model survival analysis framework
* a clinicogenomic integration study
* a platform for interpretable survival exploration

**This system is not:**
* a diagnostic tool
* a clinical decision system
* a substitute for clinical judgment""")

        st.markdown("---")


    # ---------- FINAL SECTION ----------
    st.header("What the Project Achieved and Why It Matters")
    st.markdown("""Using a compact, mechanistic feature set, the models achieved C-indices of approximately **0.79 (Cox)** and **~0.73 (RSF)** on TCGA test data, with performance remaining around **0.63** on the externally validated METABRIC cohort—levels considered meaningful for genomic survival modeling across cohorts.

High- and low-risk groups showed clearly separated Kaplan–Meier curves, and the same immune, metabolic, and extracellular matrix programs emerged as drivers of hazard in both datasets.

Rather than acting as a complex classifier, OncoRisk functions as a **computational risk architecture**: it translates clinicogenomic profiles into transparent survival estimates, exposes the biological programs shaping risk, and provides a stable framework for exploring survival behavior across independent patient populations.

*Developed as a research and translational modeling project.
Not intended for clinical use.*
""")

# ============================================================
# PAGE: DASHBOARD
# ============================================================

elif page == "Dashboard":
    st.write("🧪 ENTERED:", page)

    st.title("OncoRisk™ — Dual-Model Survival Analytics")
    

    art = get_artifacts()

    df_tcga = art["df_tcga"]
    df_mb = art["df_mb"]
    df_test_scaled = art["df_test"]
    df_mb_scaled = art["df_mb_s"]

    features = art["features"]
    gene_features = art["gene_features"]

    scaler = art["scaler"]
    cph = art["cph"]
    rsf = art["rsf"]

    cindex_train_cox = art["cindex_train_cox"]
    cindex_train_rsf = art["cindex_train_rsf"]

    risk_test_cox = art["risk_test_cox"]
    risk_test_rsf = art["risk_test_rsf"]

    df_tcga_test_km = art["df_tcga_test_km"]
    df_mb_km = art["df_mb_km"]

    df_imp_cox = art["df_imp_cox"]
    df_imp_rsf = art["df_imp_rsf"]

    cindex_test_cox = art["cindex_test_cox"]
    cindex_test_rsf = art["cindex_test_rsf"]
    cindex_mb_cox = art["cindex_mb_cox"]
    cindex_mb_rsf = art["cindex_mb_rsf"]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Cox Train C-index", f"{cindex_train_cox:.3f}")
    col2.metric("Cox Test C-index",  f"{cindex_test_cox:.3f}")
    col3.metric("Cox External",      f"{cindex_mb_cox:.3f}")
    col4.metric("RSF Train C-index", f"{cindex_train_rsf:.3f}")
    col5.metric("RSF Test C-index",  f"{cindex_test_rsf:.3f}")
    col6.metric("RSF External",      f"{cindex_mb_rsf:.3f}")

    st.caption("Cox = mechanistic hazard; RSF = nonlinear ensemble risk. Metrics are concordance indices.")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("CoxPH – Global Feature Importance")
        top_cox = df_imp_cox.head(25)

        fig, ax = plt.subplots(figsize=(5, 6))  # taller for 25 labels
        y_pos = np.arange(len(top_cox))
        ax.barh(y_pos, top_cox["abs_coef"].values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_cox["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("|Coefficient| (log-hazard)")
        ax.set_title("Top 25 Features by |Cox Coefficient|")
        fig.tight_layout()
        st.pyplot(fig)

    with colB:
        st.subheader("RSF – Global Feature Importance")
        if not df_imp_rsf.empty:
            top_rsf = df_imp_rsf.head(25)
            fig2, ax2 = plt.subplots(figsize=(5, 6))  # taller
            y_pos2 = np.arange(len(top_rsf))
            ax2.barh(y_pos2, top_rsf["abs_importance"].values)
            ax2.set_yticks(y_pos2)
            ax2.set_yticklabels(top_rsf["feature"].values)
            ax2.invert_yaxis()
            ax2.set_xlabel("Importance")
            ax2.set_title("Top 25 Features by RSF Importance")
            fig2.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("RSF importance not available in this build.")

# ============================================================
# PAGE: PATIENT RISK PROFILE INFERENCE
# ============================================================

elif page == "Patient Risk Profile Inference":
    st.write("🧪 ENTERED:", page)

    st.title("Patient Risk Profile Inference (Cox + RSF)")
    art = get_artifacts()

    df_tcga = art["df_tcga"]
    df_test_scaled = art["df_test"]
    df_mb_scaled = art["df_mb_s"]

    features = art["features"]
    gene_features = art["gene_features"]

    scaler = art["scaler"]
    cph = art["cph"]
    rsf = art["rsf"]

    mode = st.radio(
        "Input mode",
        ["Manual profile", "Select existing TCGA test patient"],
        horizontal=True
    )

    if mode == "Select existing TCGA test patient":
        # pick patient ID from test set
        pid = st.selectbox("Patient ID (TCGA test)", df_test_scaled.index.tolist())

        # ---- UI defaults from RAW, not scaled ----
        AGE_default = float(df_tcga.loc[pid, "AGE"])
        NODE_default = 1 if df_tcga.loc[pid, "NODE_POS"] == 1 else 0


        # ---- SAFETY: ensure dropdown index is valid ----
        if NODE_default not in [0, 1]:
            NODE_default = 0




    else:
        pid = None
        AGE_default = 55.0
        NODE_default = 0


    # -------------------------------
    # CLINICAL INPUTS
    # -------------------------------
    st.markdown("#### Clinical Inputs")
    c1, c2 = st.columns(2)
    AGE = c1.number_input("Age (years)", 0.0, 100.0, float(AGE_default))
    NODE_STR = c2.selectbox(
        "Lymph Node Status",
        ["Negative", "Positive"],
        index=NODE_default
    )
    NODE_POS = 1 if NODE_STR == "Positive" else 0

    # -------------------------------
    # GENE INPUTS
    # -------------------------------
    st.markdown("#### Molecular Profile (optional; default = 0)")
    gene_vals = {}

    with st.expander("Gene expression inputs (z-scored / approximate)"):
        for g in gene_features:
            gene_vals[g] = st.number_input(g, value=0.0, key=f"gene_{g}")

    # -------------------------------
    # RUN INFERENCE
    # -------------------------------
    if st.button("Run Risk Inference"):

        # Build raw feature row
        row_dict = {"AGE": AGE, "NODE_POS": NODE_POS}
        row_dict.update(gene_vals)
        row_raw = pd.DataFrame([row_dict])[features]

        # Scale
        X_row = scaler.transform(row_raw.values)
        df_row_scaled = pd.DataFrame(X_row, columns=features)

        # Cox hazard + survival curve
        hazard_cox = float(cph.predict_partial_hazard(df_row_scaled).values[0])

        surv_func_cox = cph.predict_survival_function(df_row_scaled)
        times_cox = surv_func_cox.index.values.astype(float)
        surv_cox = surv_func_cox.values[:, 0].astype(float)

        # RSF risk + survival curve
        surv_funcs_rsf = rsf.predict_survival_function(X_row)
        sf_rsf = surv_funcs_rsf[0]
        times_rsf = sf_rsf.x.astype(float)
        surv_rsf = sf_rsf.y.astype(float)

        # common grid
        t_min = 0.0
        t_max = min(times_cox.max(), times_rsf.max())
        grid = np.linspace(t_min, t_max, 200)

        # interpolate
        surv_cox_grid = np.interp(grid, times_cox, surv_cox)
        surv_rsf_grid = np.interp(grid, times_rsf, surv_rsf)

        # Survival-time metrics
        median_cox = median_survival_time(grid, surv_cox_grid)
        median_rsf = median_survival_time(grid, surv_rsf_grid)
        rmst_cox = rmst(grid, surv_cox_grid)
        rmst_rsf = rmst(grid, surv_rsf_grid)

        consensus_median = np.nanmean([median_cox, median_rsf])
        agree = agreement_score(median_cox, median_rsf)
        agree_label = agreement_label(agree)

        col1, col2, col3 = st.columns(3)
        col1.metric("Cox Hazard (relative)", f"{hazard_cox:.2f}")
        col2.metric("RSF Risk (cum. hazard)", f"{sf_rsf.y[-1]:.2f}")
        col3.metric(
            "Model Agreement",
            f"{agree:.2f}" if not np.isnan(agree) else "N/A",
            help=f"Agreement: {agree_label}"
        )

        # Survival Time Estimates
        st.markdown("#### Survival Time Estimates")
        c4, c5, c6 = st.columns(3)

        if np.isnan(median_cox):
            c4.metric("Cox Median Survival", "Not reached")
        else:
            c4.metric("Cox Median Survival (days)", f"{median_cox:.1f}")

        if np.isnan(median_rsf):
            c5.metric("RSF Median Survival", "Not reached")
        else:
            c5.metric("RSF Median Survival (days)", f"{median_rsf:.1f}")

        if np.isnan(consensus_median):
            c6.metric("Consensus Median", "Not reached")
        else:
            c6.metric("Consensus Median (days)", f"{consensus_median:.1f}")

        st.caption("If survival never falls below 0.5, median survival is reported as 'Not reached'.")

        # RMST
        st.markdown("#### Restricted Mean Survival Time (RMST)")
        c7, c8 = st.columns(2)
        c7.metric("Cox RMST (days)", f"{rmst_cox:.1f}")
        c8.metric("RSF RMST (days)", f"{rmst_rsf:.1f}")

        # Plot curves
        st.markdown("#### Survival Curves (Cox vs RSF)")
        fig, ax = plt.subplots(figsize=(5, 4))

        ax.plot(grid, surv_cox_grid, label="CoxPH", linestyle="-")
        ax.plot(grid, surv_rsf_grid, label="RSF", linestyle="--")

        if not np.isnan(median_cox):
            ax.axvline(median_cox, linestyle=":", alpha=0.7)

        if not np.isnan(median_rsf):
            ax.axvline(median_rsf, linestyle=":", alpha=0.7)

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# PAGE: COHORT ANALYTICS (OPTION B)
# ============================================================

elif page == "Cohort Analytics":
    st.write("🧪 ENTERED:", page)

    st.title("Cohort-Level Survival Analytics")
    art = get_artifacts()

    df_tcga_test_km = art["df_tcga_test_km"]
    df_mb_km = art["df_mb_km"]

    risk_test_cox = art["risk_test_cox"]
    risk_test_rsf = art["risk_test_rsf"]

    km = KaplanMeierFitter()
   
    # -------------------------
    # 1. KM Curves – TCGA
    # -------------------------
    st.markdown("### 1. KM Curves – TCGA Test (Cox-based stratification)")
    df_km = df_tcga_test_km.copy()

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    for g in ["Low", "High"]:
        sub = df_km[df_km["group_cox"] == g]
        km.fit(sub["time"], sub["event"], label=g)
        km.plot_survival_function(ax=ax1)

    ax1.set_title("TCGA Test – KM by Cox Median Hazard Split")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Survival probability")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # -------------------------
    # 2. KM Curves – METABRIC
    # -------------------------
    st.markdown("### 2. KM Curves – METABRIC External (Cox-based stratification)")
    df_km_mb = df_mb_km.copy()

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    for g in ["Low", "High"]:
        sub = df_km_mb[df_km_mb["group_cox"] == g]
        km.fit(sub["time"], sub["event"], label=g)
        km.plot_survival_function(ax=ax2)

    ax2.set_title("METABRIC – KM by Cox Median Hazard Split")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Survival probability")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # -------------------------
    # 3. Risk Distributions
    # -------------------------
    st.markdown("### 3. Risk Distributions – TCGA Test (Cox vs RSF)")
    c1, c2 = st.columns(2)

    with c1:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.hist(risk_test_cox, bins=30, alpha=0.8)
        ax3.set_title("CoxPH Hazard – TCGA Test")
        ax3.set_xlabel("Hazard score")
        ax3.set_ylabel("Count")
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with c2:
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.hist(risk_test_rsf, bins=30, alpha=0.8)
        ax4.set_title("RSF Risk – TCGA Test")
        ax4.set_xlabel("Risk (cum. hazard)")
        ax4.set_ylabel("Count")
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # -------------------------
    # 4. Scatter
    # -------------------------
    st.markdown("### 4. Cox vs RSF Agreement – TCGA Test")
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    ax5.scatter(risk_test_cox, risk_test_rsf, alpha=0.4)
    ax5.set_xlabel("CoxPH Hazard")
    ax5.set_ylabel("RSF Risk")
    ax5.set_title("Twin-Model Risk Landscape – TCGA Test")
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    st.caption(
        "KM curves and risk distributions reveal cohort-level survival structure "
        "and alignment between linear and nonlinear risk modeling."
    )

# ============================================================
# PAGE: FEATURE INTELLIGENCE
# ============================================================

elif page == "Feature Intelligence":
    st.write("🧪 ENTERED:", page)

    st.title("Feature-Level Intelligence (Cox + RSF)")
    art = get_artifacts()

    df_imp_cox = art["df_imp_cox"]
    df_imp_rsf = art["df_imp_rsf"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CoxPH – Top Features")
        top_cox = df_imp_cox.head(30)

        fig, ax = plt.subplots(figsize=(5, 6))  # taller for 30 genes
        y_pos = np.arange(len(top_cox))
        ax.barh(y_pos, top_cox["coef"].values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_cox["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient (log-hazard)")
        ax.set_title("Top 30 Features by |Cox Coefficient|")
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("RSF – Top Features")
        if not df_imp_rsf.empty:
            top_rsf = df_imp_rsf.head(30)
            fig2, ax2 = plt.subplots(figsize=(5, 6))  # taller
            y_pos2 = np.arange(len(top_rsf))
            ax2.barh(y_pos2, top_rsf["importance"].values)
            ax2.set_yticks(y_pos2)
            ax2.set_yticklabels(top_rsf["feature"].values)
            ax2.invert_yaxis()
            ax2.set_xlabel("Importance")
            ax2.set_title("Top 30 Features by RSF Importance")
            fig2.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("RSF importance not available in this build.")

    st.caption("Cox gives directional effect; RSF gives nonlinear contribution. Together they form a dual lens.")

# ============================================================
# PAGE: GENE INTELLIGENCE
# ============================================================

elif page == "Gene Intelligence":
    st.write("🧪 ENTERED:", page)

    st.title("Gene-Level Biological Intelligence")
    art = get_artifacts()

    df_imp_cox = art["df_imp_cox"]

    df_genes_cox = df_imp_cox[df_imp_cox["type"] == "gene"].copy()
    df_genes_cox = df_genes_cox.sort_values("abs_coef", ascending=False)

    gene = st.selectbox("Select gene", df_genes_cox["feature"].tolist())

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Top Genes by |Cox Coefficient|")
        st.dataframe(
            df_genes_cox[["feature", "coef", "abs_coef"]].head(25),
            use_container_width=True,
            height=400
        )

    with colB:
        st.markdown("#### Gene Narrative & Metrics")

        if gene in GENE_NARRATIVES:
            st.markdown("**Biological Narrative**")
            st.write(GENE_NARRATIVES[gene])
        else:
            st.write(
                "This gene contributes to the hazard model through its coefficient magnitude and sign. "
                "Higher absolute coefficient implies stronger influence on risk."
            )

        g_row = df_genes_cox[df_genes_cox["feature"] == gene].iloc[0]
        st.markdown("**Model Contribution (CoxPH)**")
        st.write(f"- Coefficient (log-hazard): {g_row['coef']:.4f}")
        st.write(f"- |Coefficient|: {g_row['abs_coef']:.4f}")
        st.write(f"- Rank by |coef|: {df_genes_cox.index.get_loc(g_row.name) + 1}")

# ============================================================
# PAGE: MODEL FACTS
# ============================================================

elif page == "Model Facts":
    st.write("🧪 ENTERED:", page)

    st.title("Model Architecture & Performance")
    art = get_artifacts()

    df_tcga = art["df_tcga"]
    df_mb = art["df_mb"]

    cindex_test_cox = art["cindex_test_cox"]
    cindex_test_rsf = art["cindex_test_rsf"]
    cindex_mb_cox = art["cindex_mb_cox"]
    cindex_mb_rsf = art["cindex_mb_rsf"]

    st.write(f"""
**Architecture**

- Cox proportional hazards (mechanistic hazard estimation)
- Random Survival Forest (nonlinear ensemble risk)

**Cohorts**

- TCGA (n = {df_tcga.shape[0]})
- METABRIC (n = {df_mb.shape[0]})

**Feature Set**

- Clinical: AGE, lymph node status (NODE_POS)
- Molecular: curated 31-gene panel
""")

    col1, col2, col3 = st.columns(3)
    col1.metric("Cox Test C-index", f"{cindex_test_cox:.3f}")
    col2.metric("RSF Test C-index", f"{cindex_test_rsf:.3f}")
    col3.metric("External (Cox / RSF)",
                f"{cindex_mb_cox:.3f} / {cindex_mb_rsf:.3f}")

    st.markdown("---")
    st.write("""
This dual-model system is designed for:

- Transparent hazard modeling (Cox)
- Nonlinear pattern capture (RSF)
- Cohort-level profiling via KM and risk distributions
- Patient-level twin inference with survival-time estimation

Survival times are derived from full survival curves:

- For each model: S(t) over a time grid
- Median survival: first t where S(t) ≤ 0.5
- RMST: area under S(t)

Consensus survival is a simple average of Cox and RSF medians when both exist.
If neither curve crosses 0.5, median survival is reported as “not reached”.
    """)

    st.caption(
        "This platform is for research and stratification. It is not a regulated clinical decision system."
    )

else:
    st.error(f"❌ Unknown page: {repr(page)}")
