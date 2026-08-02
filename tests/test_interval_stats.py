"""
Sanity tests for interval_pca.interval_stats: degenerate (point) data
should reduce exactly to classical statistics, and interval results
must always contain the classical result computed from the interval
midpoints (inclusion monotonicity intuition).
"""
import numpy as np

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from interval_pca.interval_stats import (
    interval_standardize_column,
    interval_correlation_pair,
    interval_correlation_matrix,
)


def test_degenerate_intervals_reduce_to_classical_standardization():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s_lo, s_hi = interval_standardize_column(x, x, n_restarts=2)
    xbar = x.mean()
    sd = np.sqrt(np.mean((x - xbar) ** 2))
    expected = (x - xbar) / sd
    assert np.allclose(s_lo, expected, atol=1e-6)
    assert np.allclose(s_hi, expected, atol=1e-6)


def test_degenerate_intervals_reduce_to_classical_correlation():
    rng = np.random.default_rng(1)
    r = rng.normal(size=10)
    s = 0.6 * r + rng.normal(scale=0.3, size=10)
    iv = interval_correlation_pair(r, r, s, s, n_restarts=3)
    expected = np.corrcoef(r, s)[0, 1]
    assert abs(iv.lo - expected) < 1e-4
    assert abs(iv.hi - expected) < 1e-4


def test_interval_correlation_contains_midpoint_correlation():
    """
    The interval correlation must always contain the correlation
    computed at the interval midpoints, since the midpoint
    configuration is always a feasible point of the joint optimisation
    box (Theorem 2.1.1's inclusion-monotonicity intuition applied to
    the correlation function itself).
    """
    rng = np.random.default_rng(2)
    n = 8
    r_lo = rng.normal(size=n)
    r_hi = r_lo + rng.uniform(0.1, 1.0, size=n)
    s_lo = rng.normal(size=n)
    s_hi = s_lo + rng.uniform(0.1, 1.0, size=n)

    iv = interval_correlation_pair(r_lo, r_hi, s_lo, s_hi, n_restarts=6)
    r_mid = (r_lo + r_hi) / 2
    s_mid = (s_lo + s_hi) / 2
    mid_corr = np.corrcoef(r_mid, s_mid)[0, 1]
    assert iv.lo - 1e-6 <= mid_corr <= iv.hi + 1e-6


def test_correlation_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(3)
    n, p = 6, 3
    X_lo = rng.normal(size=(n, p))
    X_hi = X_lo + rng.uniform(0.1, 0.5, size=(n, p))
    G = interval_correlation_matrix(X_lo, X_hi, n_restarts=3)
    assert np.allclose(G.lo, G.lo.T)
    assert np.allclose(G.hi, G.hi.T)
    assert np.allclose(np.diag(G.lo), 1.0)
    assert np.allclose(np.diag(G.hi), 1.0)


def test_correlation_bounds_are_within_valid_range():
    rng = np.random.default_rng(4)
    n, p = 6, 3
    X_lo = rng.normal(size=(n, p))
    X_hi = X_lo + rng.uniform(0.1, 0.5, size=(n, p))
    G = interval_correlation_matrix(X_lo, X_hi, n_restarts=3)
    assert np.all(G.lo >= -1.0 - 1e-9)
    assert np.all(G.hi <= 1.0 + 1e-9)
    assert np.all(G.lo <= G.hi + 1e-9)
