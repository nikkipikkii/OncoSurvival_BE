import pandas as pd
import numpy as np

# Biological narratives serve as the static knowledge base.
# While the text is static, the association (which gene gets which rank) 
# is determined dynamically by the model below.
GENE_NARRATIVES = {
    "KLRB1": "Encodes CD161, a marker of killer cell lectin-like receptors. High expression is often associated with favorable prognosis in breast cancer due to tumor-infiltrating lymphocytes (TILs) and immune cytotoxicity.",
    "CCL19": "A chemokine that organizes immune responses. High expression correlates with the presence of tertiary lymphoid structures (TLS), generally indicating a robust anti-tumor immune microenvironment.",
    "CLEC3A": "C-type lectin domain family member. Its role is context-dependent but often linked to extracellular matrix interactions and structural integrity within the tumor stroma.",
    "LINC01235": "A long non-coding RNA. Elevated levels have been linked to poor outcomes in several cancers, potentially driving epithelial-to-mesenchymal transition (EMT) pathways.",
    "QPRT": "Involved in the NAD+ salvage pathway. High expression can support metabolic plasticity in tumor cells, allowing survival under metabolic stress conditions.",
    "TNFRSF14": "Herpesvirus entry mediator (HVEM). It acts as a molecular switch for T-cell activation or inhibition. Its prognostic value depends on the balance of co-stimulatory vs. co-inhibitory signals.",
    "SERPINA1": "Encodes Alpha-1 Antitrypsin. High levels can inhibit proteases that degrade the extracellular matrix, potentially limiting invasion, though its role in cancer inflammation is complex.",
    "SEMA3B": "A semaphorin often acting as a tumor suppressor by inhibiting angiogenesis and cell migration. Loss of expression is frequently observed in aggressive tumors.",
    "EDA2R": "Ectodysplasin A2 receptor. It can induce apoptosis via p53-dependent pathways. Downregulation may allow tumor cells to evade cell death mechanisms.",
    "UTP23": "Involved in ribosome biogenesis. Upregulation supports the high protein synthesis demands of rapidly proliferating cancer cells.",
    "CD24": "A sialoglycoprotein linked to cell adhesion and metastasis. High expression is a well-known marker of stemness and aggressive tumor behavior.",
    "TFPI2": "Tissue factor pathway inhibitor 2. It regulates matrix remodeling and invasion. Silencing via methylation is common in aggressive cancers."
}

def get_gene_intelligence(art):
    """
    Extracts dynamic gene rankings and coefficients from the model artifacts.
    Merges them with biological narratives for the frontend.
    """
    # 1. Retrieve the Importance Dataframe from Artifacts
    if "df_imp_cox" not in art:
        return []

    df_imp = art["df_imp_cox"]

    # 2. Filter for Genes only (excluding clinical features like Age/Stage)
    #    and Sort by Absolute Coefficient Magnitude (Model Contribution)
    df_genes = df_imp[df_imp["type"] == "gene"].copy()
    df_genes = df_genes.sort_values("abs_coef", ascending=False)

    gene_data = []

    # 3. Iterate dynamically through the top genes
    #    'rank' is generated based on the sort order from the model
    for rank, (idx, row) in enumerate(df_genes.iterrows(), 1):
        gene_name = row["feature"]
        
        # Get narrative if available, else generate a generic data-driven narrative
        narrative = GENE_NARRATIVES.get(gene_name, (
            f"This gene ({gene_name}) was identified by the Cox model as a significant predictor. "
            f"It holds a rank of #{rank} based on its absolute hazard contribution, indicating it "
            "plays a non-trivial role in the model's risk stratification logic."
        ))

        gene_data.append({
            "rank": rank,
            "gene": gene_name,
            # Rounding for clean UI display
            "coef": round(float(row["coef"]), 4),
            "abs": round(float(row["abs_coef"]), 4),
            "narrative": narrative
        })

    return gene_data