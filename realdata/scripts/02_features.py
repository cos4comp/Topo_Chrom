#!/usr/bin/env python3
"""Per-gene TCIT features at 10 kb for one cell line, mirroring Methods 4.2 of the
manuscript: window of +-30 bins (61x61, +-300 kb) centred on the TSS;
features = persistence entropies H0,H1,H2 (weighted Vietoris-Rips on the window,
dissimilarity = 1 - contact/max), contact frequency CF (window mean, diagonal
excluded), TAD enrichment exp(-dist_to_boundary_bins/10) from Rao Arrowhead calls,
GC fraction over the same window. Target: log1p(mean TPM of 2 ENCODE reps).
Usage: 02_features.py <GM12878|K562>"""
import sys, os, gzip, math, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from multiprocessing import Pool
from hic_reader import HicFile
from ripser import ripser

LINE = sys.argv[1]
RES = 10000
HALF = 30

HIC_URLS = {
    'GM12878': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_insitu_primary+replicate_combined.hic',
    'K562': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_combined.hic',
}
RNA_FILES = {
    'GM12878': ['data/ENCFF644DIQ.tsv', 'data/ENCFF456WWG.tsv'],
    'K562': ['data/ENCFF172GIN.tsv', 'data/ENCFF768TKT.tsv'],
}
TAD_FILES = {
    'GM12878': 'data/GSE63525_GM12878_primary+replicate_Arrowhead_domainlist.txt.gz',
    'K562': 'data/GSE63525_K562_Arrowhead_domainlist.txt.gz',
}

def load_expression(line):
    dfs = []
    for f in RNA_FILES[line]:
        d = pd.read_csv(f, sep='\t', usecols=['gene_id', 'TPM'])
        d['gene_id'] = d['gene_id'].str.split('.').str[0]
        dfs.append(d.set_index('gene_id')['TPM'])
    tpm = pd.concat(dfs, axis=1).mean(axis=1)
    return tpm

def load_boundaries(line):
    d = pd.read_csv(TAD_FILES[line], sep='\t')
    b = {}
    for _, r in d.iterrows():
        chrom = str(r['chr1'])
        b.setdefault(chrom, []).extend([int(r['x1']), int(r['x2'])])
    return {k: np.array(sorted(set(v))) for k, v in b.items()}

def pers_entropy(diagrams_dim):
    life = np.array([d - b for b, d in diagrams_dim if np.isfinite(d) and d > b])
    if len(life) == 0 or life.sum() == 0:
        return 0.0
    p = life / life.sum()
    return float(-(p * np.log2(p)).sum())

def gene_features(args):
    tss_bin, W, bounds = args
    out = {}
    if W is None:
        return None
    Wd = W.copy()
    np.fill_diagonal(Wd, 0.0)
    out['CF'] = float(Wd.sum() / (Wd.size - Wd.shape[0]))
    mx = W.max()
    if mx <= 0:
        return None
    D = 1.0 - W / mx
    np.fill_diagonal(D, 0.0)
    dgms = ripser(D, distance_matrix=True, maxdim=2)['dgms']
    for k in range(3):
        out[f'H{k}'] = pers_entropy(dgms[k]) if k < len(dgms) else 0.0
    center_bp = (tss_bin + 0.5) * RES
    if bounds is not None and len(bounds):
        dist_bins = np.min(np.abs(bounds - center_bp)) / RES
    else:
        dist_bins = 1e6
    out['TAD'] = float(np.exp(-dist_bins / 10.0))
    return out

def main():
    genes = pd.read_csv('data/genes_hg19.tsv', sep='\t')
    genes['gene_base'] = genes['gene_id'].str.split('.').str[0]
    tpm = load_expression(LINE)
    genes['TPM'] = genes['gene_base'].map(tpm)
    genes = genes.dropna(subset=['TPM'])
    bounds = load_boundaries(LINE)
    hic = HicFile(HIC_URLS[LINE])
    # GC content per chromosome from hg19 (computed once, cached)
    gc = pickle.load(open('data/gc_windows.pkl', 'rb'))
    rows = []
    for chrom in [str(i) for i in range(1, 23)] + ['X']:
        g = genes[genes['chrom'] == 'chr' + chrom]
        if len(g) == 0:
            continue
        cache = f'data/cache/{LINE}_chr{chrom}.npz'
        if os.path.exists(cache):
            from scipy import sparse as _sp
            M = _sp.load_npz(cache).toarray().astype(np.float32)
        else:
            M = hic.matrix(chrom, RES, 'KR')
        nbins = M.shape[0]
        jobs, meta = [], []
        for _, r in g.iterrows():
            tb = int(r['tss']) // RES
            if tb - HALF < 0 or tb + HALF >= nbins:
                continue
            W = np.array(M[tb - HALF: tb + HALF + 1, tb - HALF: tb + HALF + 1],
                         dtype=np.float32)  # explicit copy: pickling a view drags the whole chromosome matrix
            jobs.append((tb, W, bounds.get(chrom)))
            meta.append(r)
        del M
        with Pool(4) as pool:
            feats = pool.map(gene_features, jobs, chunksize=64)
        for r, f in zip(meta, feats):
            if f is None:
                continue
            rows.append(dict(gene_id=r['gene_base'], gene_name=r['gene_name'],
                             chrom=r['chrom'], tss=int(r['tss']),
                             GC=gc.get((r['chrom'], int(r['tss'])), np.nan),
                             logTPM=float(np.log1p(r['TPM'])), TPM=float(r['TPM']), **f))
        print(f'{LINE} chr{chrom}: {len(rows)} genes cumulative', flush=True)
    df = pd.DataFrame(rows).dropna(subset=['GC'])
    df.to_csv(f'data/features_{LINE}.tsv', sep='\t', index=False)
    print(f'{LINE}: {len(df)} genes -> data/features_{LINE}.tsv')

if __name__ == '__main__':
    main()
