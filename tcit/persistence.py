"""Persistent homology and persistence entropy of contact windows.

For each window of the contact matrix we build a filtered simplicial complex
and compute the persistence barcode in homology dimensions k = 0, 1, 2. The
filtration is the weighted (Vietoris-Rips) filtration constructed from the
contact weights via the inverse-contact distance transform,

    D_ij = 1 / (c_ij + epsilon),   D_ii = 0,

so that higher contact corresponds to smaller distance and therefore earlier
birth in the sublevel filtration. Persistence entropy in dimension k is

    H_pers^(k) = - sum_i (l_i / L) log2 (l_i / L),

with l_i = d_i - b_i the lifetime of the i-th homology class and L the total
lifetime (essential classes, whose death is infinite, are excluded; with m
finite classes the entropy lies in [0, log2 m]).
"""

from __future__ import annotations

import numpy as np
from ripser import ripser


def barcode_to_persistence_entropy(barcode: np.ndarray) -> float:
    """Compute persistence entropy from a persistence barcode.

    Parameters
    ----------
    barcode : (m, 2) array of [birth, death] pairs; classes with infinite
        death (essential homology classes) are excluded from the entropy.

    Returns
    -------
    entropy : float, bits
    """
    barcode = np.asarray(barcode, dtype=float)
    finite = barcode[np.isfinite(barcode[:, 1])]
    if finite.size == 0:
        return 0.0
    lifetimes = finite[:, 1] - finite[:, 0]
    total = float(lifetimes.sum())
    if total <= 0.0:
        return 0.0
    p = lifetimes / total
    return float(-np.sum(p * np.log2(p)))


def persistence_entropy_contact(
    window_contacts: np.ndarray,
    max_dim: int = 2,
    pseudocount: float = 0.1,
) -> dict:
    """Persistence entropy per homology dimension for one contact window.

    Parameters
    ----------
    window_contacts : (m, m) square contact sub-matrix of a genomic window.
    max_dim : maximum homology dimension to compute (0, 1, 2).
    pseudocount : epsilon added before inverting the contact weights.

    Returns
    -------
    entropies : dict {k: H_pers^(k)} for k in {0, ..., max_dim}, in bits.
    """
    c = np.asarray(window_contacts, dtype=float)
    n = c.shape[0]
    if n < 2:
        return {k: 0.0 for k in range(max_dim + 1)}
    distance = np.zeros((n, n))
    off = ~np.eye(n, dtype=bool)
    distance[off] = 1.0 / (c[off] + pseudocount)
    result = ripser(distance, maxdim=max_dim, coeff=2, do_cocycles=False,
                    distance_matrix=True)
    entropies = {}
    for k in range(max_dim + 1):
        entropies[k] = barcode_to_persistence_entropy(result["dgms"][k])
    return entropies


def persistence_barcodes(window_contacts: np.ndarray, max_dim: int = 2,
                         pseudocount: float = 0.1):
    """Return the raw persistence barcodes for one contact window.

    Useful for plotting barcode panels and diagnostics.
    """
    c = np.asarray(window_contacts, dtype=float)
    n = c.shape[0]
    distance = np.zeros((n, n))
    off = ~np.eye(n, dtype=bool)
    distance[off] = 1.0 / (c[off] + pseudocount)
    return ripser(distance, maxdim=max_dim, coeff=2, do_cocycles=False,
                  distance_matrix=True)["dgms"]
