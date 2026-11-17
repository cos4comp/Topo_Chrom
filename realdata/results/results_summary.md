# Real-data analysis — measured results (2026-11-03)

Design: Rao et al. Hi-C (GSE63525, hg19) at 10 kb, KR normalization; per-gene
features in a ±300 kb window (61×61 bins) around the TSS, mirroring Methods 4.2:
persistence entropies H0/H1/H2 (Vietoris–Rips on dissimilarity 1 − W/max W),
contact frequency CF (window mean, diagonal excluded), TAD enrichment
exp(−dist-to-nearest-Arrowhead-boundary / 10 bins), GC fraction over the window.
Target: log1p(mean TPM), ENCODE polyA RNA-seq, 2 reps per line
(GM12878 ENCSR000AEF; K562 ENCSR000CPH). Genes: GENCODE v29 protein-coding,
TSS lifted GRCh38→hg19 (CrossMap, 19,859/19,940 = 99.6%).
Model: OLS. Validation: whole-chromosome holdout (train odd autosomes,
test even autosomes, chrX excluded).

## Genes analysed
- GM12878: 19,449 genes (train 10,571 / test 8,036 in holdout)
- K562: 18,691 genes (train 9,813 / test 8,036)

## Whole-chromosome holdout (test-set metrics)
| Line | Feature set | R² | Pearson | Spearman |
|---|---|---|---|---|
| GM12878 | GC only | −0.002 | 0.040 | 0.076 |
| GM12878 | Topology (H0,H1,H2,CF,TAD) | 0.074 | 0.277 | 0.304 |
| GM12878 | Full (topology + GC) | 0.081 | 0.290 | 0.316 |
| K562 | GC only | 0.013 | 0.120 | 0.151 |
| K562 | Topology | 0.070 | 0.268 | 0.273 |
| K562 | Full | 0.070 | 0.267 | 0.273 |

## Cross-cell-line transfer (full model, shared genes)
| Direction | R² | Pearson | Spearman |
|---|---|---|---|
| GM12878 → K562 | −0.194 | 0.190 | 0.189 |
| K562 → GM12878 | 0.029 | 0.193 | 0.221 |

## Univariate Spearman vs logTPM
| Feature | GM12878 | K562 |
|---|---|---|
| H0 | −0.175 | −0.202 |
| H1 | −0.025 | −0.247 |
| H2 | −0.105 | −0.248 |
| CF | +0.144 | +0.136 |
| TAD | +0.311 | +0.203 |
| GC | +0.085 | +0.174 |

## Standardized coefficients (full model, per 1 SD of feature)
- GM12878: TAD +0.471, GC −0.153, CF +0.118, H0 −0.103, H1 −0.037, H2 −0.011
- K562: TAD +0.211, H1 −0.214, H2 −0.184, H0 −0.117, CF −0.061, GC +0.009

## Sensitivity: expressed genes only (TPM > 1), full model, same holdout
- GM12878: R² = 0.017 (topology-only 0.010)
- K562: R² = 0.014 (topology-only 0.011)

## Honest reading
- The TCIT topology features carry genuine out-of-sample signal on real data:
  they beat the GC-only baseline by ~6–9× in holdout R² (0.070–0.081 vs −0.002–0.013)
  under a leakage-free whole-chromosome split.
- The signal is mostly domain-activity state: it largely separates expressed
  from silent genes. Among already-expressed genes, graded expression is only
  weakly predicted (R² ≈ 0.01–0.02). Claims must be scoped accordingly
  (10 kb features resolve domain-level activity, not fine per-gene levels).
- Cross-line transfer is weak (Pearson ≈ 0.19; one direction negative R²):
  feature scales and expression programmes are partly line-specific; the
  within-line model does not transport unchanged.
- H0 persistence entropy is near its ceiling (log2 60 ≈ 5.91 bits; observed
  SD 0.004 in GM12878) yet still ranks genes (ρ ≈ −0.18/−0.20): its dynamic
  range at this window size is compressed and it should not be over-interpreted.
- δ = I(G;P|T) is NOT estimated: these data have no genotype variation.
  δ estimation requires genotype-linked cohorts and remains future work.
