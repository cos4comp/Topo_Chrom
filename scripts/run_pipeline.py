"""Main analysis pipeline (merged TCIT implementation).

    python -m scripts.run_pipeline

Generates the synthetic Hi-C contact matrix and planted TAD boundaries
(manuscript-calibrated parameters), computes per-gene topological features
(persistence entropy H0/H1/H2, contact frequency, TAD enrichment), constructs
a ground-truth expression from a linear generative model on standardized
features, runs the OLS + partial-correlation + bootstrap analysis (Table 1:
five predictors), and performs the maximum-entropy inference check. All
inputs/outputs are written under data/ and results/.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcit.synthetic_hic import SyntheticHiC
from tcit.features import (compute_features, pick_gene_bins,
                           ground_truth_expression)
from tcit.statistics import (partial_correlations, regression_table,
                             bootstrap_r2, bootstrap_ci)
from tcit.maxent import run_maxent_demo

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_config():
    with open(os.path.join(ROOT, "config", "generative_parameters.json")) as f:
        return json.load(f)


def main():
    cfg = load_config()

    # ------------------------------------------------------------------
    # 1. Synthetic Hi-C contact matrix
    # ------------------------------------------------------------------
    print("[1/6] Generating synthetic Hi-C contact matrix ...")
    synth = SyntheticHiC(cfg)
    contact, boundaries = synth.generate()
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    np.save(os.path.join(ROOT, "data", "contact_matrix.npy"), contact)
    np.save(os.path.join(ROOT, "data", "tad_boundaries.npy"), boundaries)
    print(f"  bins={contact.shape[0]}, planted TAD boundaries={len(boundaries)}")

    # ------------------------------------------------------------------
    # 2. Per-gene topological features
    # ------------------------------------------------------------------
    print("[2/6] Computing persistence features ...")
    fp = cfg["features"]
    eg = cfg["expression_ground_truth"]
    gene_bins = pick_gene_bins(contact, int(eg["n_genes"]),
                               seed=int(fp["seed"]),
                               window=int(fp["window_size_bins"]))
    feats = compute_features(
        contact,
        boundaries,
        gene_bins,
        window=int(fp["window_size_bins"]),
        max_dim=int(fp["persistence_max_dim"]),
        pseudocount=float(fp["pseudocount"]),
        chain_link_distance=float(fp["chain_link_distance"]),
        tad_decay_length=float(fp["tad_decay_length"]),
    )
    feats.to_csv(os.path.join(ROOT, "data", "features.csv"), index=False)
    print(f"  n_genes={len(feats)}, window={fp['window_size_bins']} bins")

    # ------------------------------------------------------------------
    # 3. Ground-truth expression (standardized predictors, target R^2)
    # ------------------------------------------------------------------
    print("[3/6] Constructing ground-truth expression ...")
    y, lp, sigma, Xstd = ground_truth_expression(feats, eg)
    feats["expression"] = y
    feats.to_csv(os.path.join(ROOT, "data", "features.csv"), index=False)
    print(f"  noise sigma={sigma:.3f}, target R^2={eg['target_r2']}")

    # ------------------------------------------------------------------
    # 4. Regression + partial correlations + bootstrap (Table 1)
    # ------------------------------------------------------------------
    print("[4/6] Regression analysis ...")
    st = cfg["statistics"]
    h1 = feats["H1"].to_numpy(dtype=float)
    h2 = feats["H2"].to_numpy(dtype=float)
    Xfull = np.column_stack([Xstd, h1, h2])
    names = [
        "Persistence entropy (H0)",
        "Contact frequency",
        "TAD enrichment",
        "Persistence entropy (H1)",
        "Persistence entropy (H2)",
    ]
    partial_r = partial_correlations(Xfull, y)
    table, res = regression_table(Xfull, y, names, partial_r)
    r2s = bootstrap_r2(Xfull, y, n_boot=int(st["n_bootstrap"]),
                       seed=int(st["seed"]))
    lo, hi = bootstrap_ci(r2s)
    table_out = table.copy()
    table_out.loc[len(table_out)] = {
        "feature": "R^2 (95% CI)",
        "coefficient": round(res["r2"], 3),
        "se": "",
        "beta_std": "",
        "partial_r": "",
        "t": "",
        "p_value": f"[{lo:.3f}, {hi:.3f}]",
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    table_out.to_csv(os.path.join(ROOT, "results", "table1_regression.csv"),
                     index=False)
    np.save(os.path.join(ROOT, "results", "bootstrap_r2.npy"), r2s)
    print("  R^2 =", round(res["r2"], 4), f"95% CI [{lo:.4f}, {hi:.4f}]")
    for _, row in table.iterrows():
        print(f"  {row['feature']:<28} beta_std={row['beta_std']:>6.3f} "
              f"partial_r={row['partial_r']:>6.3f} p={row['p_value']}")

    # ------------------------------------------------------------------
    # 5. Maximum-entropy inference check
    # ------------------------------------------------------------------
    print("[5/6] Maximum-entropy inference ...")
    mx = cfg["maxent"]
    n_loci = int(mx["n_loci"])
    # choose the first planted TAD with at least n_loci bins
    chosen = None
    start = 0
    for b in np.append(boundaries, contact.shape[0]):
        if b - start >= n_loci:
            chosen = contact[start:b, start:b]
            break
        start = b
    if chosen is None:
        chosen = contact[:n_loci, :n_loci]
    me = run_maxent_demo(chosen, mx)
    with open(os.path.join(ROOT, "results", "maxent_inference.json"), "w") as f:
        json.dump(me, f, indent=2)
    print(f"  field correlation rho(h_true,h_inf) = {me['field_correlation']:.3f}")
    print(f"  coupling w_inf/w_true             = {me['coupling_recovery_ratio']:.3f}")
    print(f"  pairwise-corr mean abs error      = {me['pairwise_corr_mean_abs_error']:.4f}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("[6/6] Done. Artifacts written to data/ and results/.")


if __name__ == "__main__":
    main()