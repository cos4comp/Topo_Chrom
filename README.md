# TCIT analysis pipeline

Companion code for the manuscript "Topological Chromatin Information
Theory" (PLOS Computational Biology, PCOMPBIOL-D-26-01280). The repository
contains two analyses:

* **Simulation pipeline** (this README; `tcit/`, `scripts/`, `data/`,
  `results/`): the calibrated-simulation recovery test of manuscript
  Sections 2.5/2.9 and Methods 4.1/4.2/4.5. Generates Hi-C-like contact
  matrices with planted TAD domains, computes per-gene persistence-entropy
  features (homology dimensions 0, 1, 2), and regresses a ground-truth
  expression on the topological features. This experiment is a
  calibration/recovery test of the implementation against a known ground
  truth — not evidence of predictive power on biological data.
* **Real-data analysis** (`realdata/`): the empirical application of
  manuscript Section 2.10 (Table 3, Methods 4.7) — TCIT features from
  measured Rao et al. Hi-C (GSE63525) in GM12878 and K562 at 10 kb
  predicting ENCODE RNA-seq expression under whole-chromosome holdout.
  See `realdata/README.md`.

The `tcit/maxent.py` module is the reference implementation of the
maximum-entropy construction of manuscript Section 2.4 / Methods 4.3
(pseudolikelihood inference, Metropolis–Hastings sampling); the manuscript
reports no numerical results from that inference.

## Repository components

| Component | Description |
| --------- | ----------- |
| `config/generative_parameters.json` | Fixed generative parameters and seeds |
| `tcit/synthetic_hic.py` | Synthetic Hi-C contact map with planted TAD domains (distance decay + enrichment + Gamma noise) |
| `tcit/persistence.py` | Persistent homology and persistence entropy (dimensions 0, 1, 2) via `ripser` |
| `tcit/features.py` | Per-gene H0/H1/H2, contact frequency, TAD enrichment |
| `tcit/maxent.py` | Maximum-entropy model, pseudolikelihood inference, Metropolis–Hastings sampling |
| `tcit/statistics.py` | OLS, partial correlations, 1,000-resample bootstrap |
| `tcit/detection.py` | Data-driven TAD boundary detection (directionality index, Dixon et al. 2012) |
| `scripts/run_pipeline.py` | End-to-end pipeline |
| `scripts/sensitivity_analysis.py` | Parameter sensitivity of the regression R² |
| `scripts/make_figure1.py` | Reproduces the overview figure |

## Quick start

Requires Python ≥ 3.9 (numpy, scipy, pandas, ripser, matplotlib).

```bash
pip install -r requirements.txt

# 1. End-to-end pipeline (generates data/ and results/)
python -m scripts.run_pipeline

# 2. Sensitivity analysis
python -m scripts.sensitivity_analysis

# 3. Reproduce the overview figure
python -m scripts.make_figure1

# 4. Run the smoke tests
python -m pytest tests/ -q
```

## What the pipeline computes

1. **Synthetic Hi-C.** Contact counts
   `c_ij = (α exp(−|i−j|/λ) + c₀)·κ(i,j)·η_ij`, where κ is the planted-TAD
   enrichment factor and η is Gamma noise (symmetrized, as real contact maps
   are symmetric).
2. **Topological features.** For each of `n_genes` genes a symmetric 15-bin
   window around the TSS bin is filtered by the inverse-contact distance
   transform and the persistence entropy in homology dimensions 0–2 is computed
   with `ripser`; contact frequency and TAD enrichment (exp(−distance to
   nearest planted boundary)) complete the feature set.
3. **Ground-truth expression.** `E = β₀ + β_H₀·H₀ + β_CF·CF + β_TAD·TAD + ε`,
   with ε scaled so the noise-free R² equals the configured target (0.72).
4. **Regression.** OLS with standard errors, standardized coefficients,
   partial correlations, and 95% bootstrap CIs (1,000 resamples).
5. **Maximum-entropy model.** The least-biased distribution over 3-state
   chromatin configurations with Hamiltonian
   `H(σ) = −Σ h_i σ_i − ½Σ J_ij σ_i σ_j`, couplings `J_ij ∝ c_ij`, is inferred
   by pseudolikelihood from Gibbs-sampled data and then sampled by
   Metropolis–Hastings to verify that target observables are reproduced.

All random seeds and parameters are fixed, so every artifact in `data/`,
`results/`, and `figures/` is reproducible from the code alone.

## Example outputs (realized values from the default parameters)

Running `python -m scripts.run_pipeline` currently produces:

* **Regression (recovery test).** Multiple-regression of the planted
  expression on the three topological features (n = 1,000 genes) recovers
  R² = **0.739** against the planted target of 0.72. Because the expression
  data are simulated from a known ground truth, hypothesis-testing
  statistics on them carry no inferential meaning: the manuscript's
  Table 1 is a recovery report (coefficients and standardized
  coefficients) and reports no p-values or partial correlations.
* **Maximum-entropy module (code sanity check only).** On Gibbs-sampled
  synthetic configurations, the inference of `tcit/maxent.py` recovers
  local fields with ρ(h, ĥ) = **0.907** and coupling scale ŵ/w = **0.996**.
  These numbers check the implementation; they are not results of the
  manuscript (see Methods 4.3).
* **Sensitivity.** R² remains within ≈0.70–0.81 across the tested ranges of
  window size (10–25 bins), contact threshold, TAD-detection threshold, and
  number of genes, and declines monotonically from 0.96 to 0.52 as the noise
  s.d. rises from 0.1 to 0.5 (see `results/sensitivity_table.csv`).

These values are specific to the placeholder parameters and will change if the
parameters are updated to match a manuscript.

## Outputs

* `data/contact_matrix.npy`, `data/tad_boundaries.npy` — generated contact map
  and planted boundaries.
* `data/features.csv` — per-gene H0/H1/H2, contact frequency, TAD enrichment,
  expression.
* `results/table1_regression.csv` — regression table (coefficients,
  standard errors, standardized coefficients; the inferential columns of
  the raw pipeline output — partial correlations and p-values on planted
  data — were removed on 2026-11-03 to match the revised manuscript's
  Table 1; see `CHANGES_2026-11-03.md`).
* `results/sensitivity_table.csv` — sensitivity table.
* `results/bootstrap_r2.npy`, `results/maxent_inference.json` — bootstrap
  distribution and maximum-entropy recovery metrics.
* `figures/topo_crom_dots.png` — overview figure.

## Project structure

```
tcit_repo/
├── config/generative_parameters.json
├── tcit/            # library modules (importable)
├── scripts/         # executable analysis scripts
├── tests/           # smoke tests
├── data/            # generated contact map + features
├── results/         # tables and inference outputs
├── figures/         # overview figure
├── realdata/        # Section 2.10 real-data analysis (scripts, data, results)
└── requirements.txt
```
