#!/usr/bin/env python3
"""Chromosome-holdout validation of TCIT features on real data.
Split: train on odd autosomes, test on even autosomes (genes grouped by whole
chromosomes, so no locus appears in both). Models: OLS on feature sets
[GC] / [topology: H0,H1,H2,CF,TAD] / [full]. Also cross-cell-line transfer.
Writes results/metrics.csv"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

TOPO = ['H0', 'H1', 'H2', 'CF', 'TAD']
FULL = TOPO + ['GC']
ODD = {f'chr{i}' for i in range(1, 23) if i % 2 == 1}
EVEN = {f'chr{i}' for i in range(1, 23) if i % 2 == 0}

def evaluate(train, test, feats):
    Xtr, ytr = train[feats].to_numpy(), train['logTPM'].to_numpy()
    Xte, yte = test[feats].to_numpy(), test['logTPM'].to_numpy()
    m = LinearRegression().fit(Xtr, ytr)
    pred = m.predict(Xte)
    return dict(r2=r2_score(yte, pred), pearson=pearsonr(yte, pred)[0],
                spearman=spearmanr(yte, pred)[0], n_test=len(yte),
                coef=dict(zip(feats, np.round(m.coef_, 4))), intercept=round(float(m.intercept_), 4))

def main():
    out = []
    lines = {}
    for line in ['GM12878', 'K562']:
        df = pd.read_csv(f'data/features_{line}.tsv', sep='\t')
        df = df[df['chrom'] != 'chrX']
        lines[line] = df
        tr, te = df[df['chrom'].isin(ODD)], df[df['chrom'].isin(EVEN)]
        for name, feats in [('GC only', ['GC']), ('Topology only', TOPO), ('Full (topology + GC)', FULL)]:
            r = evaluate(tr, te, feats)
            out.append(dict(kind=f'{line} holdout', model=name, **{k: r[k] for k in ['r2', 'pearson', 'spearman', 'n_test']}))
            print(line, name, {k: round(r[k], 4) for k in ['r2', 'pearson', 'spearman']}, 'coef:', r['coef'])
    # cross-line transfer on shared genes, full feature set
    g, k = lines['GM12878'], lines['K562']
    common = set(g['gene_id']) & set(k['gene_id'])
    g2, k2 = g[g['gene_id'].isin(common)], k[k['gene_id'].isin(common)]
    for a, b, an, bn in [(g2, k2, 'GM12878', 'K562'), (k2, g2, 'K562', 'GM12878')]:
        r = evaluate(a, b, FULL)
        out.append(dict(kind=f'transfer {an}->{bn}', model='Full (topology + GC)',
                        **{kk: r[kk] for kk in ['r2', 'pearson', 'spearman', 'n_test']}))
        print(f'transfer {an}->{bn}', {kk: round(r[kk], 4) for kk in ['r2', 'pearson', 'spearman']})
    pd.DataFrame(out).to_csv('results/metrics.csv', index=False)
    print('saved results/metrics.csv')

if __name__ == '__main__':
    main()
