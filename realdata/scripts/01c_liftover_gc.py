"""Lift hg38 TSSs (data/genes.tsv, GENCODE v29) to hg19 via CrossMap
and compute GC content in +-300 kb TSS windows from the UCSC hg19 fasta.
Outputs data/genes_hg19.tsv and data/gc_windows.pkl."""
import gzip
import pickle
import subprocess
import pandas as pd

genes = pd.read_csv('data/genes.tsv', sep='\t')
genes['base_id'] = genes['gene_id'].str.split('.').str[0]
bed = pd.DataFrame({'chrom': genes['chrom'], 'start': genes['tss'] - 1,
                    'end': genes['tss'], 'name': genes['base_id']})
bed.to_csv('tmp/tss_hg38.bed', sep='\t', header=False, index=False)
subprocess.run(['CrossMap', 'bed', 'data/hg38ToHg19.chain.gz',
                'tmp/tss_hg38.bed', 'tmp/tss_hg19.bed'], check=True)
lifted = pd.read_csv('tmp/tss_hg19.bed', sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'name'])
lifted['tss'] = lifted['end']  # point BED: end == lifted coordinate (1-based)
m = genes.merge(lifted[['name', 'chrom', 'tss']],
                left_on='base_id', right_on='name', suffixes=('_38', '_19'))
out = pd.DataFrame({'gene_id': m['base_id'], 'gene_name': m['gene_name'],
                    'chrom': m['chrom_19'], 'tss': m['tss_19'],
                    'strand': m['strand']})
out = out[out['chrom'].isin(['chr%d' % i for i in range(1, 23)] + ['chrX'])].copy()
out.to_csv('data/genes_hg19.tsv', sep='\t', index=False)
print(f'genes hg38 {len(genes)} -> lifted hg19 {len(out)} '
      f'({100.0 * len(out) / len(genes):.1f}%)')

by_chrom = {c: g for c, g in out.groupby('chrom')}
gc = {}
cur, buf = None, []
with gzip.open('data/hg19_ucsc.fa.gz', 'rt') as fh:
    for line in fh:
        if line.startswith('>'):
            if cur in by_chrom:
                seq = ''.join(buf).upper()
                for _, r in by_chrom[cur].iterrows():
                    w = seq[max(0, r['tss'] - 300_000):r['tss'] + 300_000]
                    if w:
                        gc[(cur, int(r['tss']))] = (w.count('G') + w.count('C')) / len(w)
            cur, buf = line[1:].split()[0], []
            if not cur.startswith('chr'):
                cur = 'chr' + cur
        else:
            buf.append(line.strip())
if cur in by_chrom:
    seq = ''.join(buf).upper()
    for _, r in by_chrom[cur].iterrows():
        w = seq[max(0, r['tss'] - 300_000):r['tss'] + 300_000]
        if w:
            gc[(cur, int(r['tss']))] = (w.count('G') + w.count('C')) / len(w)
with open('data/gc_windows.pkl', 'wb') as fh:
    pickle.dump(gc, fh)
print(f'GC windows: {len(gc)}')
