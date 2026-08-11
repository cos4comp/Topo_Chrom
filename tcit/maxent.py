"""Maximum-entropy (Potts) model of chromatin states with pseudolikelihood
inference.

The least-biased distribution over chromatin configurations given observed mean
states and pairwise correlations is the Boltzmann distribution with
Hamiltonian

    H(sigma) = - sum_i h_i sigma_i - 1/2 sum_{i != j} J_ij sigma_i sigma_j,

with sigma_i in {-1, 0, +1}. The pairwise couplings are taken proportional to
Hi-C contact frequencies,

    J_ij(s, t) = w * c_ij * A[s, t],

where A is a symmetric 3x3 interaction matrix (A[0, .] = A[., 0] = 0,
A[1,1] = A[2,2] = 1, A[1,2] = A[2,1] = r). The fields h and the coupling
parameters (w, r) are inferred from observed configurations by
pseudolikelihood maximization; the inferred model is then sampled with
Metropolis-Hastings and the target mean states and pairwise correlations are
checked to be reproduced. No parameter values here are quoted from a specific
manuscript; they are placeholder settings.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def potts_logits(onehot, h, w, r, contact):
    """Unnormalized log-probabilities (logits) for each site and sample.

    Parameters
    ----------
    onehot : (T, n, 3) one-hot encoded states.
    h : (n, 3) fields (h[:, 0] = 0 by gauge).
    w : scalar coupling scale.
    r : scalar cross-coupling parameter.
    contact : (n, n) contact sub-matrix of the chosen TAD.

    Returns
    -------
    logits : (T, n, 3)
    """
    T, n, q = onehot.shape
    x1 = onehot[:, :, 1]  # (T, n) indicator of state 1
    x2 = onehot[:, :, 2]  # (T, n) indicator of state 2
    # neighbor sums weighted by contact
    n1 = x1 @ contact.T  # (T, n)  sum_j c_ij [x_j == 1]
    n2 = x2 @ contact.T
    logit1 = h[:, 1][None, :] + w * (n1 + r * n2)
    logit2 = h[:, 2][None, :] + w * (r * n1 + n2)
    logits = np.zeros((T, n, q))
    logits[:, :, 1] = logit1
    logits[:, :, 2] = logit2
    return logits


def pseudolikelihood_loss(theta, onehot, contact, l2):
    """Negative average conditional log-likelihood with L2 regularization."""
    n = contact.shape[0]
    h = np.zeros((n, 3))
    h[:, 1] = theta[:n]
    h[:, 2] = theta[n:2 * n]
    w = theta[2 * n]
    r = theta[2 * n + 1]

    logits = potts_logits(onehot, h, w, r, contact)
    logsum = np.log(np.sum(np.exp(logits), axis=2))  # (T, n)
    probs = np.exp(logits) / np.exp(logsum)[:, :, None]
    # target: observed state of each site
    target = onehot
    nll = -np.sum(target * (logits - logsum[:, :, None]))
    nll /= onehot.shape[0]
    reg = l2 * 0.5 * (np.sum(h ** 2) + w ** 2 + r ** 2)
    return nll + reg


def sample_gibbs(h, w, r, contact, n_samples, burnin, seed):
    """Gibbs sampling from the target Potts model (used to generate data)."""
    rng = np.random.default_rng(seed)
    n = contact.shape[0]
    states = rng.integers(0, 3, size=n)
    samples = np.zeros((n_samples, n), dtype=int)
    for step in range(n_samples + burnin):
        for i in range(n):
            logits = np.zeros(3)
            n1 = np.sum(contact[i] * (states == 1))
            n2 = np.sum(contact[i] * (states == 2))
            logits[1] = h[i, 1] + w * (n1 + r * n2)
            logits[2] = h[i, 2] + w * (r * n1 + n2)
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            states[i] = rng.choice(3, p=p)
        if step >= burnin:
            samples[step - burnin] = states
    return samples


def sample_metropolis_hastings(h, w, r, contact, n_steps, burnin, seed):
    """Metropolis-Hastings sampling from the inferred model.

    Uses incremental single-site energy updates: the energy change on flipping
    site i from s to s' is

        dE = h_i[s'] - h_i[s] + sum_{j != i} (W_ij[s', s_j] - W_ij[s, s_j]),

    with W_ij[s, t] = w * c_ij * A[s, t] and A the 3x3 interaction matrix.
    """
    rng = np.random.default_rng(seed)
    n = contact.shape[0]
    A = np.zeros((3, 3))
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[1, 2] = A[2, 1] = r
    # W[i, j] is a 3x3 interaction block
    W = w * contact[:, :, None, None] * A[None, None, :, :]

    state = rng.integers(0, 3, size=n)
    h_arr = h

    def energy(s):
        e = float(np.sum(h_arr[np.arange(n), s]))
        for i in range(n):
            for j in range(i + 1, n):
                e += float(W[i, j, s[i], s[j]])
        return e

    samples = np.zeros((n_steps, n), dtype=int)
    ar = np.arange(n)
    cur = state.copy()
    cur_e = energy(cur)
    accepted = 0
    for step in range(n_steps + burnin):
        i = rng.integers(0, n)
        prop = cur.copy()
        prop[i] = rng.integers(0, 3)
        dE = (h_arr[i, prop[i]] - h_arr[i, cur[i]]
              + float(np.sum(W[i, ar, prop[i], cur] - W[i, ar, cur[i], cur])))
        if rng.random() < np.exp(dE):
            cur = prop; cur_e += dE; accepted += 1
        if step >= burnin:
            samples[step - burnin] = cur
    return samples, accepted / (n_steps + burnin)


def fit_pseudolikelihood(onehot, contact, l2=1e-3, seed=11):
    """Infer (h, w, r) from observed configurations by pseudolikelihood."""
    n = contact.shape[0]
    rng = np.random.default_rng(seed)
    theta0 = np.concatenate([rng.normal(0, 0.1, size=2 * n), [0.3, 0.0]])
    res = minimize(
        pseudolikelihood_loss,
        theta0,
        args=(onehot, contact, l2),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-8},
    )
    if not res.success:
        print("  [maxent] pseudolikelihood warning:", res.message)
    h = np.zeros((n, 3))
    h[:, 1] = res.x[:n]
    h[:, 2] = res.x[n:2 * n]
    w = res.x[2 * n]
    r = res.x[2 * n + 1]
    return h, w, r, res.fun


def run_maxent_demo(contact_tad, params):
    """End-to-end recovery test: true model -> data -> pseudolikelihood -> MCMC.

    Parameters
    ----------
    contact_tad : (m, m) contact sub-matrix of a TAD (m >= n_loci).
    params : the "maxent" section of generative_parameters.json.
    """
    n = int(params["n_loci"])
    n_samples = int(params["n_samples"])
    w_true = params["coupling_scale"]
    r_true = params["cross_coupling"] if "cross_coupling" in params else 0.3
    field_scale = params["field_scale"]
    seed = params["seed"]

    contact = contact_tad[:n, :n].astype(float)
    contact = (contact + contact.T) / 2.0
    np.fill_diagonal(contact, 0.0)
    off = ~np.eye(n, dtype=bool)
    contact = contact / contact[off].mean()  # normalize so w is dimensionless

    rng = np.random.default_rng(seed)
    h_true = np.zeros((n, 3))
    h_true[:, 1] = rng.normal(0, field_scale, n)
    h_true[:, 2] = rng.normal(0, field_scale, n)

    states = sample_gibbs(h_true, w_true, r_true, contact,
                          n_samples, burnin=1000, seed=seed)
    onehot = np.eye(3, dtype=float)[states]  # (T, n, 3)

    h_inf, w_inf, r_inf, _ = fit_pseudolikelihood(
        onehot, contact, l2=params["pseudolikelihood_l2"], seed=seed
    )

    rho_h = float(np.corrcoef(h_true[:, 1:].ravel(),
                              h_inf[:, 1:].ravel())[0, 1])
    rho_w = float(min(abs(w_inf / w_true), 1.0) if w_true != 0 else np.nan)

    # MCMC check that the inferred model reproduces target observables
    mcmc_steps = int(params["mcmc_steps"])
    burnin = int(params["mcmc_burnin"])
    samples_mh, acc = sample_metropolis_hastings(
        h_inf, w_inf, r_inf, contact, mcmc_steps, burnin, seed=seed + 1
    )
    obs_true = states.mean(axis=0)
    obs_mcmc = samples_mh.mean(axis=0)

    # pairwise-correlation reproduction on loci that vary in both samples
    var_t = np.std(states, axis=0) > 1e-9
    var_m = np.std(samples_mh, axis=0) > 1e-9
    keep = var_t & var_m
    if keep.sum() >= 2:
        corr_true = np.corrcoef(states[:, keep].T)
        corr_mcmc = np.corrcoef(samples_mh[:, keep].T)
        err = float(np.mean(np.abs(corr_true - corr_mcmc)))
    else:
        err = float("nan")

    return {
        "n_loci": n,
        "w_true": float(w_true),
        "w_inferred": float(w_inf),
        "r_true": float(r_true),
        "r_inferred": float(r_inf),
        "field_correlation": float(rho_h),
        "coupling_recovery_ratio": float(rho_w),
        "mcmc_acceptance_rate": float(acc),
        "mean_state_max_abs_error": float(np.max(np.abs(obs_true - obs_mcmc))),
        "pairwise_corr_mean_abs_error": err,
    }
