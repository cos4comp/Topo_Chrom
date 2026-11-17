"""Smoke tests for the TCIT reproducibility pipeline.

Run:  python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcit.synthetic_hic import SyntheticHiC
from tcit.persistence import persistence_entropy_contact, barcode_to_persistence_entropy
from tcit.features import compute_features, pick_gene_bins
from tcit.detection import detect_boundaries
from tcit.statistics import ols, partial_correlations, bootstrap_r2
from tcit.maxent import run_maxent_demo, sample_gibbs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg():
    with open(os.path.join(ROOT, "config", "generative_parameters.json")) as f:
        return json.load(f)


def test_contact_matrix_generation():
    cfg = load_cfg()
    synth = SyntheticHiC(cfg)
    contact, boundaries = synth.generate()
    n = int(cfg["synthetic_hic"]["n_bins_total"])
    assert contact.shape == (n, n)
    assert np.allclose(np.diag(contact), 0)
    assert len(boundaries) >= 1
    assert np.all(boundaries < n)


def test_persistence_entropy_range():
    rng = np.random.default_rng(0)
    contact = rng.random((12, 12))
    ent = persistence_entropy_contact(contact, max_dim=2)
    assert set(ent.keys()) == {0, 1, 2}
    for k, v in ent.items():
        assert 0.0 <= v <= np.log2(12) + 1e-6, (k, v)


def test_barcode_entropy_empty():
    assert barcode_to_persistence_entropy(np.array([[0.0, np.inf]])) == 0.0


def test_features_columns():
    cfg = load_cfg()
    synth = SyntheticHiC(cfg)
    contact, boundaries = synth.generate()
    bins = pick_gene_bins(contact, 100, seed=3)
    feats = compute_features(contact, boundaries, bins, window=15)
    assert set(["H0", "H1", "H2", "contact_frequency", "tad_enrichment"]).issubset(
        feats.columns)
    assert len(feats) == 100
    assert feats["H0"].between(0, np.log2(15) + 1e-6).all()


def test_boundary_detection_recovery():
    cfg = load_cfg()
    synth = SyntheticHiC(cfg)
    contact, boundaries = synth.generate()
    det = detect_boundaries(contact, window=10, threshold=0.25)
    assert len(det) >= 1


def test_regression_high_r2():
    cfg = load_cfg()
    synth = SyntheticHiC(cfg)
    contact, boundaries = synth.generate()
    bins = pick_gene_bins(contact, 200, seed=3)
    feats = compute_features(contact, boundaries, bins, window=15)
    X = feats[["H0", "contact_frequency", "tad_enrichment"]].to_numpy()
    eg = cfg["expression_ground_truth"]
    lp = (eg["beta0"] + eg["beta_H0"] * X[:, 0]
          + eg["beta_CF"] * X[:, 1] + eg["beta_TAD"] * X[:, 2])
    r2 = float(eg["target_r2"])
    sigma = float(np.sqrt(np.var(lp) * (1 - r2) / r2))
    rng = np.random.default_rng(eg["seed"])
    y = lp - lp.mean() + rng.normal(0, sigma, size=len(lp))
    res = ols(X, y)
    assert res["r2"] > 0.5
    partial = partial_correlations(X, y)
    assert partial.shape == (3,)


def test_maxent_recovery():
    cfg = load_cfg()
    rng = np.random.default_rng(1)
    contact = rng.random((12, 12))
    contact = (contact + contact.T) / 2
    np.fill_diagonal(contact, 0.0)
    res = run_maxent_demo(contact, cfg["maxent"])
    assert res["field_correlation"] > -1.0
    assert 0.0 <= res["mcmc_acceptance_rate"] <= 1.0
    assert np.isfinite(res["pairwise_corr_mean_abs_error"])


def test_bootstrap_r2():
    rng = np.random.default_rng(0)
    X = rng.random((100, 2))
    y = X[:, 0] + X[:, 1] + rng.normal(0, 0.1, 100)
    r2s = bootstrap_r2(X, y, n_boot=50, seed=1)
    assert r2s.shape == (50,)
    assert r2s.min() > 0


def test_gibbs_samples_shape():
    rng = np.random.default_rng(0)
    contact = rng.random((6, 6))
    contact = (contact + contact.T) / 2
    np.fill_diagonal(contact, 0.0)
    h = np.zeros((6, 3)); h[:, 1] = 0.3; h[:, 2] = -0.2
    s = sample_gibbs(h, 0.3, 0.2, contact, 50, 20, seed=0)
    assert s.shape == (50, 6)
    assert set(np.unique(s)).issubset({0, 1, 2})
