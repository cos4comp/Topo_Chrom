"""Sensitivity analysis of the regression R^2.

    python -m scripts.sensitivity_analysis

The residual standard deviation sigma is fixed at its nominal value (the value
that yields the configured target R^2, 0.72, on the baseline configuration);
each parameter is then varied over the configured range while the others are
held at their nominal values, and the realized regression R^2 is recorded.
Any change in R^2 therefore reflects genuine sensitivity of the association to
that parameter, not re-fitting of the noise level.

Parameters:

    window_size_bins   : 10-25 bins
    contact_threshold  : 0.05-0.95 (floor applied to contact counts)
    tad_threshold      : 0.05-0.30 (DI boundary-detection threshold)
    n_genes            : 500-2000
    noise_sigma        : 0.1-0.5 (residual s.d. on the centered expression)

Output: results/sensitivity_table.csv
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcit.features import compute_features, pick_gene_bins
from tcit.statistics import ols
from tcit.detection import detect_boundaries

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_data():
    contact = np.load(os.path.join(ROOT, "data", "contact_matrix.npy"))
    boundaries = np.load(os.path.join(ROOT, "data", "tad_boundaries.npy"))
    with open(os.path.join(ROOT, "config", "generative_parameters.json")) as f:
        cfg = json.load(f)
    return contact, boundaries, cfg


def linear_predictor(cfg, feats):
    eg = cfg["expression_ground_truth"]
    X = feats[["H0", "contact_frequency", "tad_enrichment"]].to_numpy()
    lp = eg["beta0"] + eg["beta_H0"] * X[:, 0] + eg["beta_CF"] * X[:, 1] \
        + eg["beta_TAD"] * X[:, 2]
    return lp - lp.mean()


def feature_matrix(cfg, contact, boundaries, gene_bins, window,
                   contact_threshold, tad_threshold):
    fp = cfg["features"]
    c = contact if contact_threshold is None else np.maximum(contact, contact_threshold)
    b = boundaries if tad_threshold is None else detect_boundaries(
        c, window=10, threshold=tad_threshold)
    feats = compute_features(
        c, b, gene_bins,
        window=int(window),
        max_dim=int(fp["persistence_max_dim"]),
        pseudocount=float(fp["pseudocount"]),
    )
    return feats


def main():
    contact, boundaries, cfg = load_data()
    fp = cfg["features"]
    eg = cfg["expression_ground_truth"]
    sens = cfg["sensitivity"]
    seed = fp["seed"]

    nominal = {
        "window": int(fp["window_size_bins"]),
        "contact_threshold": None,
        "tad_threshold": None,
        "noise_sigma": None,
    }
    gene_bins = pick_gene_bins(contact, int(eg["n_genes"]), seed=seed)
    lp_nom = linear_predictor(cfg, feature_matrix(
        cfg, contact, boundaries, gene_bins,
        nominal["window"], None, None))
    target_r2 = float(eg["target_r2"])
    sigma_base = float(np.sqrt(np.var(lp_nom) * (1.0 - target_r2) / target_r2))
    rng = np.random.default_rng(eg["seed"])
    y_nom = lp_nom + rng.normal(0, sigma_base, size=len(lp_nom))
    r2_nominal = ols(feature_matrix(cfg, contact, boundaries, gene_bins,
                                    nominal["window"], None, None)
                     [["H0", "contact_frequency", "tad_enrichment"]].to_numpy(),
                     y_nom)["r2"]

    def r2_for(gb, window, contact_threshold, tad_threshold, noise_sigma):
        feats = feature_matrix(cfg, contact, boundaries, gb, window,
                               contact_threshold, tad_threshold)
        lp = linear_predictor(cfg, feats)
        y = lp + rng.normal(0, sigma_base, size=len(lp))
        if noise_sigma is not None:
            y = lp + rng.normal(0, noise_sigma, size=len(lp))
        X = feats[["H0", "contact_frequency", "tad_enrichment"]].to_numpy()
        return ols(X, y)["r2"]

    lo_w, hi_w = sens["window_size_range"]
    lo_c, hi_c = sens["contact_threshold_range"]
    lo_t, hi_t = sens["tad_threshold_range"]
    lo_n, hi_n = sens["n_genes_range"]
    lo_s, hi_s = sens["noise_sigma_range"]

    rows = [{
        "parameter": "baseline",
        "value_low": "-", "r2_low": round(r2_nominal, 3),
        "value_high": "-", "r2_high": round(r2_nominal, 3),
    }]

    def add(name, lo, hi, r2lo, r2hi):
        rows.append({"parameter": name, "value_low": lo, "r2_low": round(r2lo, 3),
                     "value_high": hi, "r2_high": round(r2hi, 3)})

    add("window_size_bins", lo_w, hi_w,
        r2_for(gene_bins, lo_w, None, None, None),
        r2_for(gene_bins, hi_w, None, None, None))
    add("contact_threshold", lo_c, hi_c,
        r2_for(gene_bins, nominal["window"], lo_c, None, None),
        r2_for(gene_bins, nominal["window"], hi_c, None, None))
    add("tad_threshold", lo_t, hi_t,
        r2_for(gene_bins, nominal["window"], None, lo_t, None),
        r2_for(gene_bins, nominal["window"], None, hi_t, None))
    gb_lo = pick_gene_bins(contact, int(lo_n), seed=seed)
    gb_hi = pick_gene_bins(contact, int(hi_n), seed=seed)
    add("n_genes", int(lo_n), int(hi_n),
        r2_for(gb_lo, nominal["window"], None, None, None),
        r2_for(gb_hi, nominal["window"], None, None, None))
    add("noise_sigma", lo_s, hi_s,
        r2_for(gene_bins, nominal["window"], None, None, lo_s),
        r2_for(gene_bins, nominal["window"], None, None, hi_s))

    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "results", "sensitivity_table.csv"), index=False)
    print(f"nominal sigma = {sigma_base:.3f}, nominal R^2 = {r2_nominal:.3f}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
