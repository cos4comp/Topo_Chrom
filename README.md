# TCIT analysis pipeline

Simulation-based analysis of topological features of synthetic chromatin
contact maps, implementing **Topological Chromatin Information Theory (TCIT)**.
The analysis covers synthetic Hi-C generation, per-gene persistence-entropy
features, ground-truth expression modeling, regression statistics, a Potts
maximum-entropy model, sensitivity analysis, and a manuscript Figure 1.

The parameters are calibrated to chromosome-1 statistics of GSE63525 (Rao et
al. 2014) and ENCODE RNA-seq (ENCSR000AEN), in line with the manuscript
submitted to *PLOS Computational Biology* (PCOMPBIOL-D-26-01280).

Every analysis step lives behind the `tcit/` library, and all scripts consume
that single implementation.

## Design of the codebase

| Concern | Where it lives |
| ------- | -------------- |
| Configuration (manuscript-calibrated params + seeds) | `config/generative_parameters.json` |
| Synthetic Hi-C generation (α=2, λ=8, c₀=0.01, κ=3, Gamma shape 2) | `tcit/synthetic_hic.py` |
| Persistence entropy H0/H1/H2 (chain-link + ε distance transform) | `tcit/persistence.py` |
| Per-gene features + ground-truth expression (`exp(−d/τ)`, z-scored predictors) | `tcit/features.py` |
| Regression statistics (OLS SE/t/p, beta_std, partial r, bootstrap) | `tcit/statistics.py` |
| Maximum-entropy Potts model (pseudolikelihood + Metropolis–Hastings) | `tcit/maxent.py` |
| Directionality-index TAD detection | `tcit/detection.py` |
| End-to-end pipeline → `data/`, `results/` CSVs | `scripts/run_pipeline.py` |
| Sensitivity analysis → `results/sensitivity_table.csv` | `scripts/sensitivity_analysis.py` |
| Manuscript Figure-1 rendering (six panels, 300 dpi) | `scripts/regression_and_plot.py` |

`run_pipeline.py` owns all statistical analysis and table output (through
`tcit/statistics.py`), and `regression_and_plot.py` only loads those artifacts
and renders the figure.

## Quick start

Requires Python ≥ 3.9 (numpy, scipy, pandas, ripser, matplotlib).

```bash
pip install -r requirements.txt

# 1. End-to-end pipeline (generates data/ and results/)
python -m scripts.run_pipeline

# 2. Sensitivity analysis
python -m scripts.sensitivity_analysis

# 3. Reproduce the manuscript figure (Figure 1)
python -m scripts.regression_and_plot

# 4. Run the smoke tests
python -m pytest tests/ -q
```

## What the pipeline computes

1. **Synthetic Hi-C.** Contact counts
   `c_ij = (α exp(−|i−j|/λ) + c₀)·κ(i,j)·η_ij`, where κ is the planted-TAD
   enrichment factor (κ = 3 inside a TAD) and η is unit-mean Gamma noise,
   symmetrized (real contact maps are symmetric).
2. **Topological features.** For each of `n_genes` genes a symmetric 15-bin
   window around the TSS bin is filtered by the inverse-contact distance
   transform
   `D_ij = 1/(c_ij + ε)` with a polymer chain-link cap on neighbouring bins,
   and the persistence entropy in homology dimensions 0–2 is computed with
   `ripser`; contact frequency, and TAD enrichment `exp(−d/τ)` (τ = 3 bins)
   complete the feature set.
3. **Ground-truth expression.** `E = β₀ + β_H₀·z(H0) + β_CF·z(CF) + β_TAD·z(TAD) + ε`
   on z-scored features, with ε scaled so the noise-free R² equals the
   configured target (0.72).
4. **Regression (Table 1).** OLS of expression on all five topological
   predictors with standard errors, standardized coefficients, partial
   correlations, and 95% bootstrap CIs (1,000 resamples).
5. **Maximum-entropy model.** The least-biased distribution over 3-state
   chromatin configurations with Hamiltonian
   `H(σ) = −Σ h_i σ_i − ½Σ J_ij σ_i σ_j`, couplings `J_ij ∝ c_ij`, is inferred
   by pseudolikelihood from Gibbs-sampled data and then sampled by
   Metropolis–Hastings to verify that target observables are reproduced.

All random seeds and parameters are fixed, so every artifact in `data/`,
`results/`, and `figures/` is reproducible from the code alone.

## Outputs

* `data/contact_matrix.npy`, `data/tad_boundaries.npy` — generated contact map
  and planted boundaries.
* `data/features.csv` — per-gene H0/H1/H2, contact frequency, TAD enrichment,
  expression.
* `results/table1_regression.csv` — five-predictor regression table.
* `results/sensitivity_table.csv` — sensitivity table.
* `results/bootstrap_r2.npy`, `results/maxent_inference.json` — bootstrap
  distribution and maximum-entropy recovery metrics.
* `figures/topo_crom.png` — manuscript Figure 1 (six-panel overview).

## Project structure

```
tcit_merged_pipeline/
├── config/generative_parameters.json
├── tcit/            # library modules (importable)
├── scripts/         # executable analysis scripts
├── tests/           # smoke tests
├── data/            # generated contact map + features
├── results/         # tables and inference outputs
├── figures/         # manuscript figure (Figure 1)
└── requirements.txt
```