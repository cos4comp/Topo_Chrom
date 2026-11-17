"""Main analysis pipeline.

    python -m scripts.run_pipeline

Generates the synthetic Hi-C contact matrix and planted TAD boundaries,
computes per-gene topological features (persistence entropy H0/H1/H2, contact
frequency, TAD enrichment), constructs a ground-truth expression from a linear
generative model, runs the OLS + partial-correlation + bootstrap analysis, and
performs the maximum-entropy inference check. All inputs/outputs are written
under data/ and results/.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcit.synthetic_hic import SyntheticHiC
from tcit.features import compute_features, pick_gene_bins
from tcit.statistics import partial_correlations, regression_table, bootstrap_r2, bootstrap_ci
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
    gene_bins = pick_gene_bins(contact, int(eg["n_genes"]), seed=fp["seed"])
    feats = compute_features(
        contact,
        boundaries,
        gene_bins,
        window=int(fp["window_size_bins"]),
        max_dim=int(fp["persistence_max_dim"]),
        pseudocount=float(fp["pseudocount"]),
    )
    feats.to_csv(os.path.join(ROOT, "data", "features.csv"), index=False)

    # ------------------------------------------------------------------
    # 3. Ground-truth expression
    # ------------------------------------------------------------------
    print("[3/6] Constructing ground-truth expression ...")
    X = feats[["H0", "contact_frequency", "tad_enrichment"]].to_numpy()
    H0, CF, TAD = X[:, 0], X[:, 1], X[:, 2]
    lp = eg["beta0"] + eg["beta_H0"] * H0 + eg["beta_CF"] * CF + eg["beta_TAD"] * TAD
    target_r2 = float(eg["target_r2"])
    sigma = float(np.sqrt(np.var(lp) * (1.0 - target_r2) / target_r2))
    rng = np.random.default_rng(eg["seed"])
    y = lp + rng.normal(0, sigma, size=len(lp))
    y_mean = y.mean()
    y = y - y_mean  # center expression
    feats["expression"] = y
    feats.to_csv(os.path.join(ROOT, "data", "features.csv"), index=False)
    print(f"  n_genes={len(y)}, noise sigma={sigma:.3f}")

    # ------------------------------------------------------------------
    # 4. Regression + partial correlations + bootstrap
    # ------------------------------------------------------------------
    print("[4/6] Regression analysis ...")
    st = cfg["statistics"]
    Xfull = X.copy()
    yfull = y.copy()
    partial_r = partial_correlations(Xfull, yfull)
    table, res = regression_table(
        Xfull, yfull, ["H0", "contact_frequency", "tad_enrichment"], partial_r
    )
    r2s = bootstrap_r2(Xfull, yfull, n_boot=int(st["n_bootstrap"]), seed=st["seed"])
    lo, hi = bootstrap_ci(r2s)
    # The manuscript's Table 1 is a recovery report: coefficients and
    # standardized coefficients only. Inferential columns (partial
    # correlations, t, p-values) are meaningful only for measured data and
    # are not written for this planted-data experiment (see
    # CHANGES_2026-11-03.md).
    table_out = table.drop(columns=[c for c in ["partial_r", "t", "p_value"]
                                    if c in table.columns]).copy()
    table_out.loc[len(table_out)] = {
        "feature": "R^2 (95% CI)",
        "coefficient": round(res["r2"], 3),
        "se": "",
        "beta_std": "",
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    table_out.to_csv(os.path.join(ROOT, "results", "table1_regression.csv"),
                     index=False)
    np.save(os.path.join(ROOT, "results", "bootstrap_r2.npy"), r2s)
    print("  R^2 =", round(res["r2"], 4), f"95% CI [{lo:.4f}, {hi:.4f}]")
    for _, row in table.iterrows():
        print(f"  {row['feature']:<20} beta_std={row['beta_std']:>6.3f} "
              f"coef={row['coefficient']:>6.3f}")

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
