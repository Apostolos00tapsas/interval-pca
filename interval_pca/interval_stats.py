"""
interval_stats.py
------------------
Interval standardization and interval correlation, following the
Appendix of:

    Gioia, F. & Lauro, C.N. (2006), "Principal Component Analysis on
    Interval Data", Computational Statistics 21, 343-363.

The paper defines these quantities as *ranges of a real (rational)
function as its arguments vary within their intervals*:

  * standardized component (App., eq. (3)-(4)):

        s_ir(x_1r,...,x_nr) = (x_ir - xbar_r) / sqrt( (1/n) sum_i (x_ir-xbar_r)^2 )

    and the interval standardized value is

        S_ir = [ min s_ir(x_1r,...,x_nr),  max s_ir(x_1r,...,x_nr) ]
                 over  x_ir in X_ir for all i=1..n

  * interval correlation between X_r^I and X_s^I:

        Corr(X_r^I, X_s^I) = [ min h(...), max h(...) ]
                 over  x_ir in X_ir, x_is in X_is, for all i=1..n

    where h(.) is the ordinary Pearson correlation function (eq. 1).

Both quantities are genuine *nonlinear, constrained optimisation
problems* -- the paper states this explicitly ("the computational cost
... refers to the cost of a constrained nonlinear optimisation
problem").  This module solves them numerically with multi-start
bounded local optimisation (L-BFGS-B via scipy), which is a standard,
practical way to approximate the (generally non-convex) min/max.  This
is declared honestly: the result is a numerically-obtained enclosure,
not a symbolically-certified global optimum, mirroring the paper's own
"linear/nonlinear programming" computational-cost remarks in Section 5.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from .interval_algebra import Interval, IntervalMatrix


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _multistart_bounds_optimum(func, lo, hi, sense="min", n_restarts=8, seed=0):
    """
    Minimise or maximise `func(x)` for x in the box [lo, hi] (elementwise),
    using multi-start L-BFGS-B.  Returns the best objective value found.

    This is used for every "min/max over the box of intervals" operation
    required by the paper's interval-standardisation and
    interval-correlation definitions.
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    bounds = list(zip(lo, hi))
    sign = 1.0 if sense == "min" else -1.0

    def obj(x):
        return sign * func(x)

    best = None
    starts = [ (lo + hi) / 2.0 ]  # midpoint start
    # include the 'all-lo' and 'all-hi' corners as informative starts
    starts.append(lo.copy())
    starts.append(hi.copy())
    for _ in range(max(0, n_restarts - len(starts))):
        starts.append(rng.uniform(lo, hi))

    for x0 in starts:
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
        val = sign * res.fun
        if best is None or (sense == "min" and val < best) or (sense == "max" and val > best):
            best = val
    return best


# --------------------------------------------------------------------------
# Interval standardisation  (Appendix, eq. (2)-(4))
# --------------------------------------------------------------------------

def _standardized_component(x_col: np.ndarray, i: int) -> float:
    """s_ir(x_1r,...,x_nr) for a fixed candidate vector x_col, at row i."""
    n = x_col.shape[0]
    xbar = x_col.mean()
    var = np.mean((x_col - xbar) ** 2)  # population variance, matches (1/n) sqrt in App.
    return (x_col[i] - xbar) / np.sqrt(var)


def interval_standardize_column(lo: np.ndarray, hi: np.ndarray,
                                 n_restarts: int = 8, seed: int = 0):
    """
    Given the lower/upper bounds (length n) of one interval-valued
    variable, return (S_lo, S_hi), each length n: the interval
    standardized value S_ir for every unit i, computed per eq. (4).
    """
    n = lo.shape[0]
    S_lo = np.empty(n)
    S_hi = np.empty(n)
    for i in range(n):
        f = lambda x, i=i: _standardized_component(x, i)
        S_lo[i] = _multistart_bounds_optimum(f, lo, hi, sense="min",
                                              n_restarts=n_restarts, seed=seed + i)
        S_hi[i] = _multistart_bounds_optimum(f, lo, hi, sense="max",
                                              n_restarts=n_restarts, seed=seed + 100 + i)
        if S_lo[i] > S_hi[i]:
            S_lo[i], S_hi[i] = S_hi[i], S_lo[i]
    return S_lo, S_hi


def interval_standardize(X_lo: np.ndarray, X_hi: np.ndarray,
                          n_restarts: int = 8, seed: int = 0):
    """
    Standardize every column (variable) of an interval data matrix
    X^I (n units x p interval variables), per the Appendix.

    Returns an IntervalMatrix S^I of the same shape.

    KNOWN LIMITATION (documented, not a bug): eq. (4) optimizes each
    unit's standardized value s_ir *independently* (a separate n-
    dimensional box optimisation per unit i). When one unit's raw
    interval is unusually wide relative to the rest of the sample (a
    "wrapping effect" precursor), this independence lets *every* unit
    reach the same extreme standardized value (+/- sqrt(n-1)), because
    each optimisation is free to move the *other* units to whatever
    configuration is most convenient for that one unit -- configurations
    that are mutually inconsistent across different i. The result is a
    valid enclosure per the paper's literal definition, but a very wide
    (oversized) one. This is the same phenomenon documented for
    interval_correlation_matrix; see ipca.py's discussion of why Gamma^I
    is computed directly from raw X^I rather than by chaining
    standardize -> correlate. Downstream quantities derived from S^I
    (e.g. IntervalPCA scores) inherit this effect and should be read
    with that caveat in datasets with an outlier-wide interval.
    """
    from .interval_algebra import IntervalMatrix
    n, p = X_lo.shape
    S_lo = np.empty((n, p))
    S_hi = np.empty((n, p))
    for j in range(p):
        s_lo, s_hi = interval_standardize_column(X_lo[:, j], X_hi[:, j],
                                                   n_restarts=n_restarts,
                                                   seed=seed + 1000 * j)
        S_lo[:, j] = s_lo
        S_hi[:, j] = s_hi
    return IntervalMatrix(S_lo, S_hi)


# --------------------------------------------------------------------------
# Interval correlation  (Appendix, eq. (1) extended to intervals)
# --------------------------------------------------------------------------

def _pearson_h(xr: np.ndarray, xs: np.ndarray) -> float:
    """h(x_1r,...,x_nr; x_1s,...,x_ns) = ordinary Pearson correlation."""
    xr = xr - xr.mean()
    xs = xs - xs.mean()
    denom = np.sqrt(np.sum(xr ** 2) * np.sum(xs ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(xr * xs) / denom)


def interval_correlation_pair(r_lo, r_hi, s_lo, s_hi,
                               n_restarts: int = 10, seed: int = 0) -> Interval:
    """
    Corr(X_r^I, X_s^I) = [min h, max h] as every x_ir in X_ir and every
    x_is in X_is range freely (Appendix, definition preceding eq. (1)).
    """
    n = r_lo.shape[0]
    lo = np.concatenate([r_lo, s_lo])
    hi = np.concatenate([r_hi, s_hi])

    def f(v):
        xr = v[:n]
        xs = v[n:]
        return _pearson_h(xr, xs)

    lo_val = _multistart_bounds_optimum(f, lo, hi, sense="min",
                                         n_restarts=n_restarts, seed=seed)
    hi_val = _multistart_bounds_optimum(f, lo, hi, sense="max",
                                         n_restarts=n_restarts, seed=seed + 1)
    lo_val = max(-1.0, min(1.0, lo_val))
    hi_val = max(-1.0, min(1.0, hi_val))
    if lo_val > hi_val:
        lo_val, hi_val = hi_val, lo_val
    return Interval(lo_val, hi_val)


def interval_correlation_matrix(X_lo: np.ndarray, X_hi: np.ndarray,
                                 n_restarts: int = 10, seed: int = 0) -> IntervalMatrix:
    """
    Compute the full p x p interval correlation matrix Gamma^I of an
    interval data matrix X^I (n x p), per the Appendix formula.
    The diagonal is exactly [1, 1].
    """
    n, p = X_lo.shape
    G_lo = np.eye(p)
    G_hi = np.eye(p)
    for r in range(p):
        for s in range(r + 1, p):
            iv = interval_correlation_pair(
                X_lo[:, r], X_hi[:, r], X_lo[:, s], X_hi[:, s],
                n_restarts=n_restarts, seed=seed + 17 * r + 31 * s
            )
            G_lo[r, s] = G_lo[s, r] = iv.lo
            G_hi[r, s] = G_hi[s, r] = iv.hi
    return IntervalMatrix(G_lo, G_hi)
