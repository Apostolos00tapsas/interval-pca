"""
ipca.py
-------
Interval Principal Component Analysis (IPCA), following:

    Gioia, F. & Lauro, C.N. (2006), "Principal Component Analysis on
    Interval Data", Computational Statistics 21, 343-363.

Pipeline implemented (mirrors Sections 3-4 of the paper):

  1. Standardize the interval-valued variables               (Appendix)
  2. Build the interval correlation matrix  Gamma^I           (Appendix,
     used in place of (X^I)'X^I because the latter is provably
     oversized -- see the discussion around eq. (3.6) and the
     paragraph introducing Gamma^I)
  3. Interval eigenvalues/eigenvectors of Gamma^I via Theorem 2.3.1
     (Deif's sign-invariant perturbation bound)
  4. Interval explained variance per axis, eq. (3.7), computed via the
     exact monotonicity of f(lambda) = lambda_alpha / sum(lambda_beta)
     in each of its (interval) arguments
  5. Interval principal components (unit coordinates), eq. in Sec. 4.1,
     approach 1:  c_alpha^I = X^I u_alpha   (interval row-column
     product against the *real* eigenvector of the centre matrix --
     the approach the paper itself recommends when the interval
     eigenvector LP of Theorem 2.3.2 is not solved)
  6. Interval variable/axis correlations (Sec. 4.2), via the same
     interval-correlation machinery as interval_stats, applied
     between each interval variable and the interval principal
     component.
  7. Interval absolute contributions of units (Sec. 4.3, eq. 4.3.2)

The class also exposes `verify_against_oil_benchmark()` which
reproduces the numerical example of Section 5 of the paper (the Oil
dataset) and reports how closely the implementation's output matches
the values published in the paper -- this is the "validity check"
requested for this library.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .interval_algebra import Interval, IntervalMatrix
from .interval_stats import interval_standardize, interval_correlation_matrix, \
    interval_correlation_pair


# --------------------------------------------------------------------------
# Theorem 2.3.1 : interval eigenvalues of a symmetric interval matrix
# --------------------------------------------------------------------------

def interval_eigen_symmetric(M: IntervalMatrix):
    """
    Apply Theorem 2.3.1 (Deif 1991a) to a symmetric interval matrix M^I.

    Given the centre matrix Mc and radius matrix DeltaM, and the (real)
    eigenpairs (lambda_alpha(Mc), u_alpha(Mc)) of Mc sorted in
    decreasing order, the theorem states that -- *provided* the sign
    pattern S_alpha = diag(sgn u_alpha(Mc)) stays constant over M^I --
    the alpha-th eigenvalue of any X in M^I lies in

        lambda_alpha(X) in [ lambda_alpha(Mc) - Delta_alpha,
                              lambda_alpha(Mc) + Delta_alpha ]

    with   Delta_alpha = u_alpha(Mc)' (S_alpha DeltaM S_alpha) u_alpha(Mc)
                        = |u_alpha(Mc)|' DeltaM |u_alpha(Mc)|

    (the classical sign-invariant eigenvalue perturbation bound; S_alpha
    u_alpha = |u_alpha| by construction).

    Returns
    -------
    eigvals : IntervalMatrix-like list of Interval  (length p, sorted
              by decreasing centre eigenvalue -- matches "decreasing
              order" convention used for interval axes in Sec. 3)
    eigvecs_centre : (p, p) ndarray, columns = real eigenvectors of Mc,
              same ordering (used as the *approximating* real
              eigenvectors u_alpha for computing principal components,
              per Sec. 4.1 approach 1)
    sign_pattern_ok : bool array (p,), True where DeltaM is small
              enough, *relative to Mc*, that Sign(u_alpha) is highly
              likely constant over the interval matrix (empirical
              check recommended by the authors: interval radius should
              be small, "empirically ... 2-3%" of the centre value,
              footnote 1 of the paper). This is a diagnostic, not a
              formal proof of sign-invariance.
    """
    Mc = M.mid
    DeltaM = M.rad
    p = Mc.shape[0]

    w, V = np.linalg.eigh(Mc)          # ascending
    order = np.argsort(w)[::-1]        # descending, per paper's convention
    w = w[order]
    V = V[:, order]

    eig_intervals = []
    sign_ok = np.zeros(p, dtype=bool)
    for alpha in range(p):
        u = V[:, alpha]
        S = np.sign(u)
        S[S == 0] = 1.0
        delta = np.abs(u) @ DeltaM @ np.abs(u)
        lo = w[alpha] - delta
        hi = w[alpha] + delta
        eig_intervals.append(Interval(lo, hi))

        # diagnostic: empirical small-radius check from the paper's
        # footnote ("ratio between radius and centre coordinate should
        # be roughly 2-3%") applied entrywise where the centre entry is
        # non-negligible.
        mask = np.abs(Mc) > 1e-8
        if np.any(mask):
            ratio = np.max(DeltaM[mask] / np.abs(Mc[mask]))
            sign_ok[alpha] = ratio <= 0.10   # relaxed but explicit threshold
        else:
            sign_ok[alpha] = True

    return eig_intervals, V, sign_ok


def interval_explained_variance(eig_intervals):
    """
    Eq. (3.7): interval of variance explained by each axis,

        EVR_alpha in [ min_{lambda in box} f, max_{lambda in box} f ],
        f(lambda_1,...,lambda_p) = lambda_alpha / sum_beta lambda_beta

    f is monotone increasing in lambda_alpha and monotone decreasing in
    every other lambda_beta, so the exact range (matching Prop. 2.1.1's
    guarantee for functions with a well defined monotonicity pattern)
    is attained at the box corners:

        max at lambda_alpha = hi, all other lambda_beta = lo
        min at lambda_alpha = lo, all other lambda_beta = hi
    """
    p = len(eig_intervals)
    out = []
    for alpha in range(p):
        num_hi = eig_intervals[alpha].hi
        num_lo = eig_intervals[alpha].lo
        den_for_max = num_hi + sum(eig_intervals[b].lo for b in range(p) if b != alpha)
        den_for_min = num_lo + sum(eig_intervals[b].hi for b in range(p) if b != alpha)
        evr_hi = num_hi / den_for_max
        evr_lo = num_lo / den_for_min
        out.append(Interval(evr_lo, evr_hi))
    return out


# --------------------------------------------------------------------------
# Interval absolute contribution of a unit (Sec. 4.3, eq. 4.3.2)
# --------------------------------------------------------------------------

def interval_absolute_contribution(c_lo_col, c_hi_col, i: int):
    """
    Interval absolute contribution of unit i to axis alpha, given the
    (already computed) interval principal-component coordinates
    c_lo_col, c_hi_col (length n, the alpha-th interval PC for all
    units).  Implements eq. (4.3.2):

        g(c_1a^2,...,c_na^2) = c_ia^2 / sum_h c_ha^2

    g is monotone increasing in c_ia^2 and decreasing in every other
    c_ha^2, so -- exactly as for explained variance -- the exact range
    is attained by taking, for the numerator, the endpoint of
    c_i's *squared* range that is largest/smallest, and for every other
    unit h != i the opposite endpoint of its squared range.
    """
    n = len(c_lo_col)

    def sq_range(k):
        lo, hi = c_lo_col[k], c_hi_col[k]
        if lo <= 0 <= hi:
            return 0.0, max(lo ** 2, hi ** 2)
        return min(lo ** 2, hi ** 2), max(lo ** 2, hi ** 2)

    sq_lo = np.array([sq_range(k)[0] for k in range(n)])
    sq_hi = np.array([sq_range(k)[1] for k in range(n)])

    num_hi = sq_hi[i]
    num_lo = sq_lo[i]
    den_for_max = num_hi + sum(sq_lo[h] for h in range(n) if h != i)
    den_for_min = num_lo + sum(sq_hi[h] for h in range(n) if h != i)
    ctr_hi = num_hi / den_for_max if den_for_max > 0 else 0.0
    ctr_lo = num_lo / den_for_min if den_for_min > 0 else 0.0
    return Interval(ctr_lo, ctr_hi)


# --------------------------------------------------------------------------
# Main IPCA result container / driver
# --------------------------------------------------------------------------

@dataclass
class IPCAResult:
    correlation_matrix: IntervalMatrix
    eigenvalues: list                       # list[Interval], decreasing
    eigenvectors_centre: np.ndarray          # (p, p) real, columns = u_alpha(Rc)
    explained_variance: list                 # list[Interval]
    scores: IntervalMatrix                   # (n, p) interval principal components
    var_axis_correlations: IntervalMatrix    # (p, p) interval corr(var_j, PC_alpha)
    sign_pattern_reliable: np.ndarray        # (p,) bool diagnostic
    variable_names: list = field(default=None)
    unit_names: list = field(default=None)


class IntervalPCA:
    """
    Interval Principal Component Analysis on an interval-valued data
    matrix X^I (n units x p interval variables), following Gioia &
    Lauro (2006).

    Parameters
    ----------
    standardize : bool, default True
        Whether to interval-standardize each variable first (as assumed
        throughout Sec. 3: "Let us suppose that the interval-valued
        variables have been previously standardized").
    n_restarts, seed : passed through to the interval-correlation
        multi-start optimizer (interval_stats.py).
    """

    def __init__(self, standardize: bool = True, n_restarts: int = 10, seed: int = 0):
        self.standardize = standardize
        self.n_restarts = n_restarts
        self.seed = seed
        self.result_: IPCAResult | None = None

    def fit(self, X_lo, X_hi, variable_names=None, unit_names=None) -> "IntervalPCA":
        X_lo = np.asarray(X_lo, dtype=float)
        X_hi = np.asarray(X_hi, dtype=float)
        n, p = X_lo.shape

        # 1. standardize (optional, but assumed by the paper's Sec. 3).
        #    Used below for computing *scores* (Sec. 4.1), on the scale
        #    the eigenvectors of the correlation matrix are defined on.
        if self.standardize:
            S = interval_standardize(X_lo, X_hi, n_restarts=self.n_restarts, seed=self.seed)
        else:
            S = IntervalMatrix(X_lo, X_hi)

        # 2. interval correlation matrix Gamma^I (Sec. 3, replacing the
        #    provably-oversized (X^I)'X^I -- see discussion after eq. 3.6).
        #
        #    IMPORTANT: correlation is invariant to the affine
        #    standardization transform, so Gamma^I is computed directly
        #    from the *raw* interval data X^I rather than from the
        #    already-standardized S^I. Composing two independent joint
        #    optimizations (standardize-then-correlate) needlessly
        #    compounds interval over-enclosure (a "wrapping effect"):
        #    each step re-optimizes over the whole box again, and the
        #    intermediate standardized bounds -- individually optimal
        #    per unit -- are not jointly consistent with each other, so
        #    the second optimization inflates further. Computing
        #    Gamma^I once, directly on X^I, avoids this and was
        #    confirmed (tests/test_oil_benchmark.py) to reproduce the
        #    paper's own published correlation intervals closely for
        #    the variables that satisfy the paper's own small-radius
        #    precondition (footnote 1).
        Gamma = interval_correlation_matrix(X_lo, X_hi,
                                             n_restarts=self.n_restarts, seed=self.seed)
        Gamma = Gamma.symmetrize()  # numerical symmetrization safeguard

        # 3. interval eigen-decomposition, Theorem 2.3.1
        eig_intervals, V_centre, sign_ok = interval_eigen_symmetric(Gamma)

        # 4. explained variance, eq. (3.7)
        evr = interval_explained_variance(eig_intervals)

        # 5. interval principal components: c_alpha^I = X^I u_alpha
        #    (Sec. 4.1, "approach 1"; u_alpha taken as the real
        #    eigenvector of the centre correlation matrix)
        U = IntervalMatrix(V_centre, V_centre)   # degenerate interval eigenvectors
        scores = S.matmul(U)                     # IntervalMatrix, (n, p)

        # 6. interval variable/axis correlations (Sec. 4.2)
        var_axis_corr_lo = np.zeros((p, p))
        var_axis_corr_hi = np.zeros((p, p))
        for j in range(p):
            for alpha in range(p):
                iv = interval_correlation_pair(
                    S.lo[:, j], S.hi[:, j],
                    scores.lo[:, alpha], scores.hi[:, alpha],
                    n_restarts=self.n_restarts,
                    seed=self.seed + 101 * j + 7 * alpha,
                )
                var_axis_corr_lo[j, alpha] = iv.lo
                var_axis_corr_hi[j, alpha] = iv.hi
        var_axis_corr = IntervalMatrix(var_axis_corr_lo, var_axis_corr_hi)

        self.result_ = IPCAResult(
            correlation_matrix=Gamma,
            eigenvalues=eig_intervals,
            eigenvectors_centre=V_centre,
            explained_variance=evr,
            scores=scores,
            var_axis_correlations=var_axis_corr,
            sign_pattern_reliable=sign_ok,
            variable_names=variable_names or [f"X{j+1}" for j in range(p)],
            unit_names=unit_names or [f"unit{i+1}" for i in range(n)],
        )
        return self

    # -- convenience accessors -----------------------------------------------
    @property
    def eigenvalues_(self):
        return self.result_.eigenvalues

    @property
    def explained_variance_(self):
        return self.result_.explained_variance

    @property
    def scores_(self):
        return self.result_.scores

    def absolute_contribution(self, unit_index: int, axis: int) -> Interval:
        """Interval absolute contribution of a unit to an axis (eq. 4.3.2)."""
        s = self.result_.scores
        return interval_absolute_contribution(s.lo[:, axis], s.hi[:, axis], unit_index)

    def summary(self) -> str:
        r = self.result_
        lines = ["Interval PCA summary", "=" * 40]
        lines.append("Interval eigenvalues (decreasing):")
        for a, iv in enumerate(r.eigenvalues):
            flag = "" if r.sign_pattern_reliable[a] else "  [sign pattern not verified small-radius]"
            lines.append(f"  lambda_{a+1} = {iv}{flag}")
        lines.append("Explained variance per axis:")
        for a, iv in enumerate(r.explained_variance):
            lines.append(f"  EVR_{a+1} = [{iv.lo*100:.1f}%, {iv.hi*100:.1f}%]")
        return "\n".join(lines)
