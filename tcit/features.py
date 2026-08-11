"""Topological feature computation for genomic windows.

For each gene we compute, within a symmetric window of ``window_size_bins``
bins centered on the bin containing the transcription start site:

* local persistence entropy in homology dimensions 0, 1, 2;
* local contact frequency: mean contact probability within the window,
  excluding the diagonal;
* TAD enrichment: exp(-d / tau) with d the distance (in bins) from the center
  bin to the nearest planted TAD boundary and tau the TAD-decay length
  (default 3 bins, ~120 kb at 40 kb resolution; manuscript convention).

The ground-truth expression used for the regression is generated on
*standardized* H0 / contact-frequency / TAD-enrichment features with Gaussian
noise scaled to hit the configured target R^2 (manuscript convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .persistence import persistence_entropy_contact


def _window_slice(center: int, window: int, n_bins: int):
    half = window // 2
    i0 = max(0, center - half)
    i1 = min(n_bins, center + half + 1)
    return slice(i0, i1)


def compute_features(
    contact: np.ndarray,
    boundaries: np.ndarray,
    gene_bins: np.ndarray,
    window: int = 15,
    max_dim: int = 2,
    pseudocount: float = 1e-5,
    chain_link_distance: float = 0.2,
    tad_decay_length: float = 3.0,
) -> pd.DataFrame:
    """Compute per-gene topological features from a synthetic contact matrix.

    Parameters
    ----------
    contact : (n_bins, n_bins) contact matrix.
    boundaries : planted TAD boundary positions.
    gene_bins : array of bin indices to which each gene's TSS is mapped.
    window : window size in bins (default 15).
    max_dim : maximum homology dimension.
    pseudocount : epsilon for the inverse-contact distance transform.
    chain_link_distance : optional cap on the distance between neighbouring
        bins that enforces a local polymer chain and stabilizes H1/H2;
        ``None`` disables it.
    tad_decay_length : decay length (bins) of the exponential TAD-enrichment
        score.

    Returns
    -------
    DataFrame with columns:
        gene, bin, H0, H1, H2, contact_frequency, tad_enrichment
    """
    n_bins = contact.shape[0]
    rows = []
    for gene, center in enumerate(gene_bins):
        sl = _window_slice(int(center), int(window), n_bins)
        sub = contact[sl, sl]
        m = sub.shape[0]
        off = ~np.eye(m, dtype=bool)
        cf = float(sub[off].mean())
        dist_boundary = float(np.min(np.abs(boundaries - int(center))))
        tad = float(np.exp(-dist_boundary / tad_decay_length))
        ent = persistence_entropy_contact(
            sub, max_dim=max_dim, pseudocount=pseudocount,
            chain_link_distance=chain_link_distance,
        )
        rows.append({
            "gene": gene,
            "bin": int(center),
            "H0": ent[0],
            "H1": ent[1],
            "H2": ent[2],
            "contact_frequency": cf,
            "tad_enrichment": tad,
        })
    return pd.DataFrame(rows)


def pick_gene_bins(contact: np.ndarray, n_genes: int, seed: int = 42,
                   window: int = 15) -> np.ndarray:
    """Sample the bin (TSS) positions of n_genes across the linear genome.

    Genes are placed strictly inside ``[window // 2, n_bins - window // 2)`` so
    that a full window around the TSS bin can always be extracted.
    """
    n_bins = contact.shape[0]
    rng = np.random.default_rng(seed)
    half = int(window) // 2
    lo, hi = half, n_bins - half
    return rng.integers(lo, hi, size=n_genes)


def ground_truth_expression(feats: pd.DataFrame, eg: dict):
    """Ground-truth expression from standardized topological features.

    E = beta0 + beta_H0 * z(H0) + beta_CF * z(CF) + beta_TAD * z(TAD) + eps,
    with z(.) the z-score and eps a Gaussian with s.d. chosen so that the
    noise-free linear predictor has variance ``var / R2`` (i.e. the residual
    s.d. equals ``sqrt(var * (1 - R2) / R2)``).

    Parameters
    ----------
    feats : DataFrame with columns H0, contact_frequency, tad_enrichment.
    eg : the "expression_ground_truth" section of the config.

    Returns
    -------
    y : (n,) ground-truth expression.
    lp : (n,) noise-free linear predictor.
    sigma : float, residual standard deviation.
    Xstd : (n, 3) standardized [H0, contact_frequency, tad_enrichment].
    """
    h0 = feats["H0"].to_numpy(dtype=float)
    cf = feats["contact_frequency"].to_numpy(dtype=float)
    tad = feats["tad_enrichment"].to_numpy(dtype=float)
    h0_std = (h0 - h0.mean()) / (h0.std() + 1e-8)
    cf_std = (cf - cf.mean()) / (cf.std() + 1e-8)
    tad_std = (tad - tad.mean()) / (tad.std() + 1e-8)
    lp = (float(eg["beta0"]) + float(eg["beta_H0"]) * h0_std
          + float(eg["beta_CF"]) * cf_std + float(eg["beta_TAD"]) * tad_std)
    target_r2 = float(eg["target_r2"])
    sigma = float(np.sqrt(np.var(lp) * (1.0 - target_r2) / target_r2))
    rng = np.random.default_rng(int(eg["seed"]))
    y = lp + rng.normal(0, sigma, size=len(lp))
    return y, lp, sigma, np.column_stack([h0_std, cf_std, tad_std])