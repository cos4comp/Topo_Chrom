# Extra statistics (04_extra_stats.py)

## GM12878 (n=19449)

Univariate Spearman vs logTPM:
- H0: rho = -0.175
- H1: rho = -0.025
- H2: rho = -0.105
- CF: rho = +0.144
- TAD: rho = +0.311
- GC: rho = +0.085

Full model, all genes: train 10571, test 8036, R2 = 0.0811
Standardized coefficients: {'H0': -0.103, 'H1': -0.037, 'H2': -0.011, 'CF': 0.118, 'TAD': 0.471, 'GC': -0.153}
Expressed only (TPM>1): full R2 = 0.0168, topology R2 = 0.0101

## K562 (n=18691)

Univariate Spearman vs logTPM:
- H0: rho = -0.202
- H1: rho = -0.247
- H2: rho = -0.248
- CF: rho = +0.136
- TAD: rho = +0.203
- GC: rho = +0.174

Full model, all genes: train 9813, test 8036, R2 = 0.0699
Standardized coefficients: {'H0': -0.117, 'H1': -0.214, 'H2': -0.184, 'CF': -0.061, 'TAD': 0.211, 'GC': 0.009}
Expressed only (TPM>1): full R2 = 0.0138, topology R2 = 0.0113
