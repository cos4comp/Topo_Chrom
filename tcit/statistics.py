"""Statistical analysis for the regression of expression on features.

Implements ordinary least squares regression with standard errors, partial
correlations, and bootstrap confidence intervals (1,000 resamples by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def ols(X, y):
    """OLS with intercept.

    Parameters
    ----------
    X : (n, k) feature matrix (no intercept column).
    y : (n,) response.

    Returns
    -------
    dict with coefficients, standard errors, t-statistics, p-values,
    standardized coefficients, and R^2.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    resid = y - yhat
    rss = float(resid @ resid)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss
    dof = n - k - 1
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    pval = 2 * stats.t.sf(np.abs(tstat), dof)
    # standardized coefficients
    xstd = np.std(X, axis=0)
    ystd = np.std(y)
    beta_std = beta[1:] * xstd / ystd
    return {
        "names": ["Intercept"] + list(range(k)),
        "coef": beta,
        "se": se,
        "t": tstat,
        "p": pval,
        "beta_std": beta_std,
        "r2": r2,
        "resid": resid,
    }


def partial_correlations(X, y):
    """Partial Pearson correlation of each feature with y given the others."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    k = X.shape[1]
    out = []
    for j in range(k):
        others = np.delete(X, j, axis=1)
        r_y = _regress_resid(y, others)
        r_x = _regress_resid(X[:, j], others)
        rho = float(np.corrcoef(r_y, r_x)[0, 1])
        out.append(rho)
    return np.array(out)


def _regress_resid(y, X):
    Xd = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return y - Xd @ beta


def bootstrap_r2(X, y, n_boot=1000, seed=13):
    """Bootstrap distribution of R^2 (1,000 resamples by default)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    r2s = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r2s[b] = ols(X[idx], y[idx])["r2"]
    return r2s


def bootstrap_ci(values, alpha=0.05):
    lo = np.percentile(values, 100 * alpha / 2)
    hi = np.percentile(values, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def regression_table(X, y, feature_names, partial_r):
    """Assemble the regression table from OLS results and partial correlations."""
    res = ols(X, y)
    rows = []
    for j, name in enumerate(feature_names):
        rows.append({
            "feature": name,
            "coefficient": round(float(res["coef"][j + 1]), 3),
            "se": round(float(res["se"][j + 1]), 3),
            "beta_std": round(float(res["beta_std"][j]), 3),
            "partial_r": round(float(partial_r[j]), 3),
            "t": round(float(res["t"][j + 1]), 3),
            "p_value": _fmt_p(float(res["p"][j + 1])),
        })
    return pd.DataFrame(rows), res


def _fmt_p(p):
    if p < 1e-15:
        return "<1e-15"
    if p < 1e-10:
        return "<1e-10"
    if p < 1e-2:
        return "<0.01"
    return f"{p:.3f}"
