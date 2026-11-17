"""Cache KR-normalized 10 kb matrices per chromosome to sparse npz files,
so the (long) feature step never needs the network again.
Usage: 02a_cache.py <GM12878|K562>"""
import os
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hic_reader import HicFile  # noqa: E402

HIC_URLS = {
    'GM12878': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_insitu_primary+replicate_combined.hic',
    'K562': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_K562_combined.hic',
}
RES = 10000

LINE = sys.argv[1]
os.makedirs('data/cache', exist_ok=True)
hic = HicFile(HIC_URLS[LINE])
for chrom in [str(i) for i in range(1, 23)] + ['X']:
    out = f'data/cache/{LINE}_chr{chrom}.npz'
    if os.path.exists(out):
        print(f'{LINE} chr{chrom}: cached, skip', flush=True)
        continue
    M = hic.matrix(chrom, RES, 'KR')
    sparse.save_npz(out, sparse.coo_matrix(M))
    print(f'{LINE} chr{chrom}: {M.shape} nnz={int(np.count_nonzero(M))} -> {out}',
          flush=True)
    del M
print(f'{LINE}: cache complete')
