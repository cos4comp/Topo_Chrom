"""
Manuscript Figure 1 generator (unified TCIT implementation).

    python -m scripts.regression_and_plot

All statistics and CSV outputs are produced once by ``scripts.run_pipeline``
(``tcit/statistics.py``); this script only loads the pipeline's data/results
artifacts and renders the six-panel manuscript figure at 300 dpi:

    figures/topo_crom.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150
})


def generate_figure_1(hic_matrix, boundaries, features, log_expr, boot_r2,
                      filename="topo_crom.png", dpi=300):
    """
    Generates the six-panel manuscript Figure 1 illustrating the TCIT pipeline
    on synthetic data.
    """
    fig, axes = plt.subplots(3, 2, figsize=(11, 14))

    # Panel A: Synthetic Hi-C Contact Map
    ax_a = axes[0, 0]
    sub_start, sub_end = 80, 125
    sns.heatmap(hic_matrix[sub_start:sub_end, sub_start:sub_end], ax=ax_a,
                cmap="YlOrRd", cbar=True, square=True)
    ax_a.set_title("A: Simulated 40-kb Hi-C Map", fontweight='bold')
    ax_a.set_xlabel("Genomic Position (bins)")
    ax_a.set_ylabel("Genomic Position (bins)")

    # Panel B: Directionality Index & Detected Boundaries
    ax_b = axes[0, 1]
    n_bins = hic_matrix.shape[0]
    di = np.zeros(n_bins)
    for i in range(2, n_bins - 2):
        up = np.sum(hic_matrix[i, i+1:i+3])
        down = np.sum(hic_matrix[i, i-2:i])
        di[i] = (up - down) / (up + down + 1e-8)
    sub_di = di[sub_start:sub_end]
    ax_b.plot(np.arange(sub_start, sub_end), sub_di, color='darkblue', linewidth=1.5)
    ax_b.axhline(0, color='gray', linestyle='--')
    for b in boundaries:
        if sub_start <= b <= sub_end:
            ax_b.axvline(b, color='red', linestyle=':',
                         label='TAD Boundary' if b == boundaries[0] else "")
    ax_b.set_title("B: Directionality Index & TADs", fontweight='bold')
    ax_b.set_xlabel("Genomic Position (bins)")
    ax_b.set_ylabel("Directionality Index")
    ax_b.set_xlim(sub_start, sub_end)

    # Panel C: Persistent Homology Barcodes (H0, H1, H2) schematic
    ax_c = axes[1, 0]
    np.random.seed(42)
    y_val = 1
    for i in range(8):
        birth = 0.0
        death = np.random.uniform(0.1, 1.2) if i < 7 else 2.5
        ax_c.plot([birth, death], [y_val, y_val], color='blue', linewidth=2,
                  label='H0' if i == 0 else "")
        y_val += 1
    for i in range(3):
        birth = np.random.uniform(0.2, 0.6)
        death = birth + np.random.uniform(0.3, 0.8)
        ax_c.plot([birth, death], [y_val, y_val], color='green', linewidth=2,
                  label='H1' if i == 0 else "")
        y_val += 1
    for i in range(1):
        birth = np.random.uniform(0.5, 0.9)
        death = birth + np.random.uniform(0.1, 0.4)
        ax_c.plot([birth, death], [y_val, y_val], color='red', linewidth=2,
                  label='H2' if i == 0 else "")
        y_val += 1

    ax_c.set_title("C: Persistent Homology Barcodes", fontweight='bold')
    ax_c.set_xlabel("Filtration Threshold (distance)")
    ax_c.set_ylabel("Homology Classes")
    ax_c.set_ylim(0, y_val + 1)
    ax_c.set_yticks([])
    ax_c.legend(loc='upper right')

    # Panel D: Simulated Expression vs. Local Persistence Entropy
    ax_d = axes[1, 1]
    sns.regplot(x=features["H0_entropy"], y=log_expr, ax=ax_d,
                scatter_kws={'alpha': 0.4, 'color': 'teal'},
                line_kws={'color': 'red', 'linewidth': 1.5})
    ax_d.set_title("D: Expression vs. H0 Entropy", fontweight='bold')
    ax_d.set_xlabel("Local H0 Persistence Entropy (bits)")
    ax_d.set_ylabel("Simulated Expression (log)")

    # Panel E: Expression Stratified by TAD-Enrichment Quartiles
    ax_e = axes[2, 0]
    quartiles = np.percentile(features["tad_enrichment"], [25, 50, 75])
    groups = np.zeros_like(features["tad_enrichment"], dtype=int)
    groups[features["tad_enrichment"] > quartiles[0]] = 1
    groups[features["tad_enrichment"] > quartiles[1]] = 2
    groups[features["tad_enrichment"] > quartiles[2]] = 3

    quartile_labels = ["Q1 (Far)", "Q2", "Q3", "Q4 (Near)"]
    sns.boxplot(x=groups, y=log_expr, ax=ax_e, palette="Blues")
    ax_e.set_xticklabels(quartile_labels)
    ax_e.set_title("E: Expression by TAD Enrichment", fontweight='bold')
    ax_e.set_xlabel("TAD Enrichment Quartile")
    ax_e.set_ylabel("Simulated Expression (log)")

    # Panel F: Bootstrap Distribution of R^2
    ax_f = axes[2, 1]
    sns.histplot(boot_r2, ax=ax_f, kde=True, color='purple', bins=25)
    ax_f.axvline(np.mean(boot_r2), color='darkred', linestyle='--',
                 label=f"Mean R² = {np.mean(boot_r2):.2f}")
    ci_lower = np.percentile(boot_r2, 2.5)
    ci_upper = np.percentile(boot_r2, 97.5)
    ax_f.axvline(ci_lower, color='red', linestyle=':', label="95% CI")
    ax_f.axvline(ci_upper, color='red', linestyle=':')
    ax_f.set_title("F: Bootstrap Distribution of R²", fontweight='bold')
    ax_f.set_xlabel("R² Value")
    ax_f.set_ylabel("Count")
    ax_f.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"Figure successfully saved to {filename} (dpi={dpi})")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    hic_matrix = np.load(os.path.join(root, "data", "contact_matrix.npy"))
    boundaries = np.load(os.path.join(root, "data", "tad_boundaries.npy"))
    feats = pd.read_csv(os.path.join(root, "data", "features.csv"))
    boot_r2 = np.load(os.path.join(root, "results", "bootstrap_r2.npy"))

    features = {
        "H0_entropy": feats["H0"].to_numpy(),
        "H1_entropy": feats["H1"].to_numpy(),
        "H2_entropy": feats["H2"].to_numpy(),
        "contact_frequency": feats["contact_frequency"].to_numpy(),
        "tad_enrichment": feats["tad_enrichment"].to_numpy(),
    }
    log_expr = feats["expression"].to_numpy()

    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    generate_figure_1(hic_matrix, boundaries, features, log_expr, boot_r2,
                      filename=os.path.join(fig_dir, "topo_crom.png"))