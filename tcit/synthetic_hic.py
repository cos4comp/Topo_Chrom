"""Synthetic Hi-C contact matrix generation.

Generates a block-enriched, Hi-C-like contact matrix:

    c_ij = (alpha * exp(-|i-j|/lambda) + c0) * kappa(i,j) * eta_ij

where |i-j| is the genomic distance in bins, kappa(i,j) is the planted TAD
enrichment factor (kappa > 1 for pairs within the same planted TAD domain,
1 otherwise), and eta_ij is multiplicative Gamma noise. The matrix is
symmetrized (real contact maps are symmetric). The parameters are placeholder
choices that produce qualitatively realistic matrices; they are not calibrated
to any specific real dataset.
"""

from __future__ import annotations

import numpy as np


class SyntheticHiC:
    """Generate synthetic, TAD-structured Hi-C contact matrices."""

    def __init__(self, params: dict):
        self.p = params
        seed = self.p["synthetic_hic"]["seed"]
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def tad_boundaries(self) -> np.ndarray:
        """Plant TAD domains with sizes drawn from the configured distribution.

        Returns the array of TAD boundary positions (bin indices at which a
        TAD ends and the next begins).
        """
        p = self.p["synthetic_hic"]
        n = int(p["n_bins_total"])
        ts = p["tad_size"]
        boundaries = []
        pos = 0
        while pos < n - 1:
            size = int(self.rng.uniform(ts["min_bins"], ts["max_bins"] + 1))
            size = min(size, n - pos)
            boundaries.append(pos + size)
            pos += size
        return np.array(boundaries, dtype=int)[:-1]

    # ------------------------------------------------------------------
    def membership(self, boundaries: np.ndarray) -> np.ndarray:
        """Return for each bin the index of the TAD domain it belongs to."""
        p = self.p["synthetic_hic"]
        n = int(p["n_bins_total"])
        m = np.zeros(n, dtype=int)
        for k, b in enumerate(boundaries):
            m[b:] = k + 1
        return m

    # ------------------------------------------------------------------
    def generate(self):
        """Generate the synthetic contact matrix.

        Returns
        -------
        contact : ndarray (n_bins, n_bins), contact counts (diagonal zero)
        boundaries : ndarray, planted TAD boundary positions
        """
        p = self.p["synthetic_hic"]
        n = int(p["n_bins_total"])

        boundaries = self.tad_boundaries()
        membership = self.membership(boundaries)

        ii = np.arange(n)[:, None]
        jj = np.arange(n)[None, :]
        dist = np.abs(ii - jj)

        base = (
            p["distance_decay_amplitude"] * np.exp(-dist / p["distance_decay_length"])
            + p["background_contact"]
        )
        kappa = np.where(
            membership[:, None] == membership[None, :],
            p["intra_tad_enrichment"],
            1.0,
        )
        noise = self.rng.gamma(
            p["noise_distribution"]["shape"],
            p["noise_distribution"]["scale"],
            size=(n, n),
        )
        contact = base * kappa * noise
        contact = (contact + contact.T) / 2.0
        np.fill_diagonal(contact, 0.0)
        return contact, boundaries
