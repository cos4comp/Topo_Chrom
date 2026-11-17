"""Topological feature computation for genomic windows.

For each gene we compute, within a symmetric window of ``window_size_bins``
bins centered on the bin containing the transcription start site:

* local persistence entropy in homology dimensions 0, 1, 2;
* local contact frequency: mean contact probability within the window,
  excluding the diagonal;
* TAD enrichment: exp(-d) with d the distance (in bins) from the center bin to
  the nearest planted TAD boundary.
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
    pseudocount: float = 0.1,
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
        tad = float(np.exp(-dist_boundary))
        ent = persistence_entropy_contact(sub, max_dim=max_dim,
                                          pseudocount=pseudocount)
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


def pick_gene_bins(contact: np.ndarray, n_genes: int, seed: int = 3) -> np.ndarray:
    """Sample the bin (TSS) positions of n_genes across the linear genome."""
    n_bins = contact.shape[0]
    rng = np.random.default_rng(seed)
    lo, hi = 1, n_bins - 1
    return rng.integers(lo, hi, size=n_genes)
