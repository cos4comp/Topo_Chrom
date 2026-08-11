"""TAD boundary detection from contact maps.

Implements a data-driven directionality-index (DI) detector: a signed,
window-averaged version of the difference between upstream and downstream
contact coverage, following the directionality index of Dixon et al.,
Nature 485:376-380 (2012). A boundary is called wherever the DI sign flips
with a magnitude larger than ``threshold``.
"""

from __future__ import annotations

import numpy as np


def directionality_index(contact: np.ndarray, window: int = 10) -> np.ndarray:
    """Window-averaged upstream-minus-downstream contact coverage.

    DI(i) = (mean_{j<i,|j-i|<=w} c_ij - mean_{j>i,|j-i|<=w} c_ij)
            / (sum of both + epsilon)

    Positive DI means the locus contacts mostly upstream bins, negative mostly
    downstream bins; a sign flip marks a domain boundary.
    """
    contact = np.asarray(contact, dtype=float)
    n = contact.shape[0]
    di = np.zeros(n)
    for i in range(n):
        i0 = max(0, i - window)
        i1 = min(n, i + window + 1)
        up = contact[max(0, i0):i, i]
        dn = contact[i + 1:i1, i]
        u = float(up.sum())
        d = float(dn.sum())
        di[i] = (u - d) / (u + d + 1e-9)
    return di


def detect_boundaries(contact: np.ndarray, window: int = 10,
                      threshold: float = 0.25) -> np.ndarray:
    """Detect TAD boundaries by DI sign-flips above a magnitude threshold."""
    di = directionality_index(contact, window)
    boundaries = []
    for i in range(1, len(di) - 1):
        if di[i - 1] < 0 and di[i] >= 0 and (di[i] - di[i - 1]) > threshold:
            boundaries.append(i)
    return np.array(boundaries, dtype=int)
