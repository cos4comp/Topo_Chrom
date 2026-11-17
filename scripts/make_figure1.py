"""Reproduce the overview figure from the deposited data.

    python -m scripts.make_figure1

Panels:
    A  Dot plot of the synthetic contact matrix (subregion of 200 bins).
    B  Directionality-index curve with detected TAD boundaries.
    C  Persistence barcodes (H0, H1, H2) of one 15-bin window.
    D  Ground-truth expression vs. local persistence entropy H0 (OLS line).
    E  Expression by TAD-enrichment quartile (box plots).
    F  Bootstrap distribution of R^2 (1,000 resamples).
    G  Topological view of the contact map -> filtration -> features schematic.

Output: figures/topo_crom_dots.png
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcit.persistence import persistence_barcodes
from tcit.detection import directionality_index, detect_boundaries
from tcit.statistics import ols, bootstrap_ci

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def panel_dots(ax, contact):
    """Dot plot of the contact matrix (random subset of pairs, scaled by count)."""
    n = contact.shape[0]
    rng = np.random.default_rng(0)
    sub = contact[:200, :200]
    maxc = sub.max()
    for i in range(200):
        for j in range(i + 1, 200):
            if sub[i, j] <= 0:
                continue
            k = min(int(np.ceil(sub[i, j] / maxc * 8)), 8)
            xs = rng.uniform(j - 0.45, j + 0.45, size=k)
            ys = rng.uniform(i - 0.45, i + 0.45, size=k)
            ax.plot(xs, ys, "k.", ms=0.8, alpha=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Genomic position (40 kb bins)")
    ax.set_ylabel("Genomic position (40 kb bins)")
    ax.set_title("A  Synthetic contact map", loc="left", fontweight="bold")


def panel_di(ax, contact, boundaries, cfg):
    window = int(cfg["features"]["window_size_bins"])
    di = directionality_index(contact[:200], window=10)
    det = detect_boundaries(contact[:200], window=10, threshold=0.25)
    ax.plot(di, color="tab:blue", lw=0.8)
    for b in det:
        ax.axvline(b, color="tab:red", lw=0.6, ls="--")
    for b in boundaries[boundaries < 200]:
        ax.axvspan(b - 1, b, color="tab:orange", alpha=0.15)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Genomic position (bins)")
    ax.set_ylabel("Directionality index")
    ax.set_title("B  TAD boundary detection", loc="left", fontweight="bold")


def panel_barcodes(ax, contact, cfg):
    fp = cfg["features"]
    w = int(fp["window_size_bins"])
    c = int(contact.shape[0] // 2)
    sl = slice(c - w // 2, c + w // 2 + 1)
    dgms = persistence_barcodes(contact[sl, sl], max_dim=int(fp["persistence_max_dim"]),
                                pseudocount=float(fp["pseudocount"]))
    colors = ["tab:gray", "tab:green", "tab:orange"]
    y = 0
    for dim in [0, 1, 2]:
        bc = dgms[dim]
        for (b, d) in bc:
            if np.isfinite(d):
                ax.plot([b, d], [y, y], color=colors[dim], lw=1.5)
                ax.plot([d], [y], "|", color=colors[dim], ms=3)
            y += 1
        y += 0.5
    ax.set_xlabel("Filtration scale (inverse contact)")
    ax.set_yticks([])
    ax.set_title("C  Persistence barcodes", loc="left", fontweight="bold")


def panel_scatter(ax, feats):
    ax.scatter(feats["H0"], feats["expression"], s=2, alpha=0.4,
               color="tab:blue")
    X = np.column_stack([np.ones(len(feats)), feats["H0"].to_numpy()])
    res = ols(feats[["H0"]].to_numpy(), feats["expression"].to_numpy())
    xs = np.linspace(feats["H0"].min(), feats["H0"].max(), 100)
    ys = res["coef"][0] + res["coef"][1] * xs
    ax.plot(xs, ys, color="tab:red", lw=1.5)
    ax.set_xlabel("Local persistence entropy H0 (bits)")
    ax.set_ylabel("Ground-truth expression")
    ax.set_title(f"D  H0 predicts expression (R^2={res['r2']:.2f})",
                 loc="left", fontweight="bold")


def panel_box(ax, feats):
    q = pd.qcut(feats["tad_enrichment"], 4, labels=False)
    groups = [feats["expression"][q == i].to_numpy() for i in range(4)]
    bp = ax.boxplot(groups, widths=0.6, showfliers=False,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("tab:orange")
        patch.set_alpha(0.7)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("TAD enrichment quartile")
    ax.set_ylabel("Expression")
    ax.set_title("E  TAD enrichment vs. expression", loc="left",
                 fontweight="bold")


def panel_bootstrap(ax, r2s):
    ax.hist(r2s, bins=40, color="tab:purple", alpha=0.7, edgecolor="k")
    lo, hi = bootstrap_ci(r2s)
    ax.axvline(np.mean(r2s), color="tab:red", lw=1.5)
    ax.axvspan(lo, hi, color="tab:red", alpha=0.15)
    ax.set_xlabel("Bootstrap R^2 (1,000 resamples)")
    ax.set_ylabel("Count")
    ax.set_title(f"F  R^2 95% CI [{lo:.3f}, {hi:.3f}]",
                 loc="left", fontweight="bold")


def panel_schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    # contact window -> triangle/vertices
    box = FancyBboxPatch((0.3, 6.4), 2.4, 2.2,
                         boxstyle="round,pad=0.08", fc="#eef", ec="k")
    ax.add_patch(box)
    ax.text(1.5, 8.9, "contact\nwindow", ha="center", va="center", fontsize=8)
    for x, y in [(0.8, 7.0), (1.7, 7.2), (2.3, 7.6), (1.2, 8.0), (2.0, 8.2)]:
        ax.plot(x, y, "ko", ms=4)
    ax.text(1.5, 6.6, "D_ij = 1/(c_ij+eps)", ha="center", fontsize=7)
    # arrow
    ax.add_patch(FancyArrowPatch((3.4, 7.5), (4.6, 7.5),
                                 arrowstyle="-|>", color="k", lw=1.5))
    ax.text(4.0, 7.9, "filtration", ha="center", fontsize=8)
    # simplices
    ax.plot([5.0, 5.6, 5.3], [7.0, 7.0, 7.5], "k-", lw=1.2)
    ax.plot([5.0, 5.6, 5.3, 5.0], [7.0, 7.0, 7.5, 7.0], "b-", alpha=0.4, lw=2)
    ax.plot(5.0, 7.0, "ko", ms=4); ax.plot(5.6, 7.0, "ko", ms=4); ax.plot(5.3, 7.5, "ko", ms=4)
    ax.text(5.3, 7.9, "Vietoris-\nRips", ha="center", fontsize=8)
    # arrow
    ax.add_patch(FancyArrowPatch((6.6, 7.5), (7.8, 7.5),
                                 arrowstyle="-|>", color="k", lw=1.5))
    ax.text(7.2, 7.9, "homology", ha="center", fontsize=8)
    # barcode
    for (y0, c) in [(7.4, "gray"), (7.15, "tab:green"), (6.9, "tab:orange")]:
        ax.plot([8.0, 9.3], [y0, y0], color=c, lw=2.5)
    ax.text(8.65, 6.6, "barcodes", ha="center", fontsize=8)
    ax.text(8.65, 6.35, "H_pers entropy", ha="center", fontsize=7)
    # arrow to features
    ax.add_patch(FancyArrowPatch((8.65, 6.2), (8.65, 5.4),
                                 arrowstyle="-|>", color="k", lw=1.5))
    box2 = FancyBboxPatch((7.2, 2.2), 2.9, 2.2, boxstyle="round,pad=0.08",
                          fc="#efe", ec="k")
    ax.add_patch(box2)
    ax.text(8.65, 4.2, "H0, H1, H2\nCF, TAD", ha="center", va="center", fontsize=8)
    # left side: genome -> genes -> window
    ax.add_patch(FancyBboxPatch((0.3, 2.2), 2.4, 2.2, boxstyle="round,pad=0.08",
                                fc="#fee", ec="k"))
    ax.text(1.5, 4.2, "TADs & genes\non linear genome", ha="center", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((2.75, 3.3), (7.15, 3.3),
                                 arrowstyle="-|>", color="k", lw=1.5))
    ax.text(4.95, 3.7, "regression (R^2)", ha="center", fontsize=8)
    ax.text(5, 1.4, "G  From contact maps to topological predictors",
           ha="center", fontsize=9, fontweight="bold")


def main():
    contact = np.load(os.path.join(ROOT, "data", "contact_matrix.npy"))
    boundaries = np.load(os.path.join(ROOT, "data", "tad_boundaries.npy"))
    feats = pd.read_csv(os.path.join(ROOT, "data", "features.csv"))
    r2s = np.load(os.path.join(ROOT, "results", "bootstrap_r2.npy"))
    with open(os.path.join(ROOT, "config", "generative_parameters.json")) as f:
        cfg = json.load(f)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axa, axb, axc = axes[0]
    axd, axe, axf = axes[1]
    axg = axes[2, 0]
    for a in axes[2, 1:]:
        a.axis("off")

    panel_dots(axa, contact)
    panel_di(axb, contact, boundaries, cfg)
    panel_barcodes(axc, contact, cfg)
    panel_scatter(axd, feats)
    panel_box(axe, feats)
    panel_bootstrap(axf, r2s)
    panel_schematic(axg)

    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    out = os.path.join(ROOT, "figures", "topo_crom_dots.png")
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
