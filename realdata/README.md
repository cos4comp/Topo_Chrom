# Real-data analysis (manuscript Section 2.10, Table 3, Methods 4.7)

Applies the TCIT feature pipeline to measured data: Rao et al. Hi-C
(GEO **GSE63525**, hg19) for GM12878 and K562 at 10 kb resolution with KR
normalization, predicting ENCODE poly(A) RNA-seq expression
(GM12878: **ENCSR000AEF**; K562: **ENCSR000CPH**; two replicates each,
averaged; target log(1+TPM)).

## Layout and external inputs

The scripts use the analysis-workspace layout (`raw/` for downloaded
inputs, `processed/` for generated tables); in this deposit, the generated
tables are under `data/` and the result files under `results/`.
Deposited inputs: the two Arrowhead domain-call files and
`gc_windows.pkl` (per-gene GC fractions). External inputs to download:
the `.hic` files from GEO GSE63525 (GM12878
`GSE63525_GM12878_insitu_primary+replicate_combined.hic`; K562
`GSE63525_K562_combined.hic`); the four ENCODE RSEM gene-quantification
files ENCFF644DIQ, ENCFF456WWG (GM12878, ENCSR000AEF) and ENCFF172GIN,
ENCFF768TKT (K562, ENCSR000CPH); the GENCODE v29 GTF (for the GRCh38 gene
table input to step 1); the UCSC hg19 reference fasta and the
hg38→hg19 liftOver chain file.

## Pipeline

Python 3.12; packages: numpy, scipy, pandas, scikit-learn, ripser,
requests, CrossMap.

1. `scripts/01c_liftover_gc.py` — protein-coding genes/TSS from GENCODE v29
   (GRCh38) lifted to hg19 with CrossMap (19,859/19,940 genes, 99.6%);
   GC fraction in ±300 kb TSS windows from the UCSC hg19 reference.
   Outputs `data/genes_hg19.tsv` and `data/gc_windows.pkl` (both deposited).
2. `scripts/02a_cache.py <LINE>` — streams the KR-normalized 10 kb matrix of
   each chromosome from the distributed `.hic` file and caches it as a
   sparse `.npz`. The `.hic` files are 13–40 GB and are **not** redistributed
   here; they are read over HTTP range requests by
   `scripts/hic_reader.py`, a purpose-written reader implementing the
   Juicebox file format (handles format v7 — the GM12878 file — and v8 —
   the K562 file; only intra-chromosomal blocks within one block of the
   diagonal are fetched).
3. `scripts/02_features.py <LINE>` — per gene, in the 61×61-bin (±300 kb)
   window around the TSS bin: persistence entropies H0/H1/H2 (Ripser,
   Vietoris–Rips on dissimilarity D = 1 − W/max W), contact frequency CF
   (window mean excluding the diagonal), TAD enrichment
   exp(−distance in bins to nearest Arrowhead boundary / 10), GC.
   Outputs `data/features_<LINE>.tsv` (the tables deposited here:
   19,449 GM12878 and 18,691 K562 genes).
4. `scripts/03_model.py` — OLS with whole-chromosome holdout (train odd
   autosomes, test even autosomes, chrX excluded) for feature sets
   GC-only / topology / full, plus cross-cell-line transfer.
   Outputs `results/metrics.csv`.
5. `scripts/04_extra_stats.py` — univariate Spearman correlations,
   standardized coefficients, and the expressed-only (TPM > 1) sensitivity
   analysis. Outputs `results/extra_stats.md`.

## Results (as reported in the manuscript)

Whole-chromosome holdout R²: topology 0.074 (GM12878) / 0.070 (K562);
full model 0.081 / 0.070; GC-only baseline −0.002 / 0.013.
Cross-line transfer is weak (Pearson r ≈ 0.19 both directions;
R² −0.19 and 0.03). Restricted to expressed genes (TPM > 1), holdout R²
falls to 0.017 / 0.014: at 10 kb the features capture domain-level
activity state rather than graded expression among active genes.
The residual δ = I(G;P|T) is **not** estimated — these data contain no
genotype variation. See `results/results_summary.md` for the full
measured record.
