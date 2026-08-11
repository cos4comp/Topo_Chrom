"""Analysis pipeline for topological features of synthetic chromatin contact
maps (merged TCIT implementation).

Implements a synthetic Hi-C contact-map generator (manuscript-calibrated),
persistent-homology feature computation with persistence entropy, a
maximum-entropy Potts model with pseudolikelihood inference and
Metropolis-Hastings sampling, and regression/statistical analysis producing a
five-predictor Table 1 and bootstrap confidence intervals. Parameters are
configured in config/generative_parameters.json.
"""

__version__ = "1.0.0"
