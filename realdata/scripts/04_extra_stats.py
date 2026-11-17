"""Supplementary statistics for the real-data analysis (Section 2.10):
univariate Spearman correlations, standardized coefficients of the full
model, and the expressed-genes-only (TPM > 1) sensitivity analysis.
Run from the realdata/ directory after 02_features.py.
Writes results/extra_stats.md."""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

FEATS = ['H0', 'H1', 'H2', 'CF', 'TAD', 'GC']
TOPO = ['H0', 'H1', 'H2', 'CF', 'TAD']


def holdout(df, cols, mask=None):
    d = df if mask is None else df[mask(df)]
    num = pd.to_numeric(d['chrom'].str[3:], errors='coerce')
    tr, te = d[num % 2 == 1], d[num % 2 == 0]
    m = LinearRegression().fit(tr[cols], tr['logTPM'])
    return len(tr), len(te), r2_score(te['logTPM'], m.predict(te[cols])), m


out = []
for line in ['GM12878', 'K562']:
    df = pd.read_csv(f'data/features_{line}.tsv', sep='\t')
    out.append(f'## {line} (n={len(df)})\n')
    out.append('Univariate Spearman vs logTPM:')
    for f in FEATS:
        r, _ = stats.spearmanr(df[f], df['logTPM'])
        out.append(f'- {f}: rho = {r:+.3f}')
    ntr, nte, r2, m = holdout(df, FEATS)
    sd = df[FEATS].std()
    std = {c: round(float(b) * float(sd[c]), 3) for c, b in zip(FEATS, m.coef_)}
    out.append(f'\nFull model, all genes: train {ntr}, test {nte}, R2 = {r2:.4f}')
    out.append(f'Standardized coefficients: {std}')
    _, _, r2e, _ = holdout(df, FEATS, mask=lambda d: d['TPM'] > 1)
    _, _, r2t, _ = holdout(df, TOPO, mask=lambda d: d['TPM'] > 1)
    out.append(f'Expressed only (TPM>1): full R2 = {r2e:.4f}, topology R2 = {r2t:.4f}\n')

with open('results/extra_stats.md', 'w') as fh:
    fh.write('# Extra statistics (04_extra_stats.py)\n\n' + '\n'.join(out))
print('\n'.join(out))
