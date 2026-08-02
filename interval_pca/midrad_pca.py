"""
midrad_pca.py
--------------
"Midpoints-Midranges PCA" (MR-PCA) for interval-valued data, following:

    Palumbo, F. & Lauro, C.N. (2003), "A PCA for interval-valued data
    based on midpoints and radii", in New Developments in Psychometrics
    (Yanai et al., eds.), Springer-Verlag, Tokyo.

Why this method exists in this library
---------------------------------------
The Gioia & Lauro (2006) IPCA implemented in `ipca.py` builds an
*interval* correlation matrix by solving nonlinear joint optimization
problems (Appendix of that paper), then propagates the resulting
interval eigenvalues via Theorem 2.3.1. This library's own validation
against the Oil benchmark (see tests/test_oil_benchmark.py) showed that
this approach can become severely *oversized* when a single unit's
interval is wide relative to the rest of the sample (here:
Saponification, driven by Linseed's [118, 196] interval) -- a known,
paper-documented limitation (footnote 1 of Gioia & Lauro; also flagged
independently by Douzal-Chouakria, Billard & Diday 2011).

Palumbo & Lauro's midpoints-and-radii approach sidesteps this failure
mode *by construction*: every interval [a, b] is represented
exactly and losslessly by its midpoint x^c = (a+b)/2 and radius
x^r = (b-a)/2 (a one-to-one, purely algebraic re-parameterisation --
no min/max optimisation anywhere). PCA is then performed using only
ordinary, real-valued linear algebra (two classical eigen-decompositions
plus an orthogonal Procrustes rotation), so there is no interval
"joint optimisation over a box" step to become oversized. This module
implements that pipeline (Sections 2-3 of the paper) and validates it
against the paper's own Oil-dataset numerical example (Section 3.5,
Table 1), reproduced in tests/test_midrad_pca_benchmark.py.

Pipeline (Sections 2-3)
------------------------
1. Represent I[X] by (Xc, Xr): centre and radius matrices (Def. midpoint
   / radius, Sec. 2). Xc is mean-centred column-wise; Xr is left as is
   (subtracting a location shift should not change a unit's *width* --
   see the docstring of `fit` for the precise convention adopted, since
   the paper's own interval-subtraction definition is ambiguous once
   applied to a non-degenerate mean interval).
2. Variance-covariance matrix (eq. 4):
       VX = (1/N)(Xc'Xc) + (1/N)(Xr'Xr) + (1/N)(Xc'Xr + Xr'Xc)
3. Standardize: Sigma = diag(sqrt(diag(VX))), Z = {Xc @ Sigma^-1, Xr @ Sigma^-1}
4. Correlation matrix (eq. 5): R2 = Zc'Zc + Zr'Zr + Zc'Zr + Zr'Zc  (trace = p)
5. Midpoints PCA (eq. 6): eigen-decomposition of Zc'Zc  ->  (lambda^c, u^c)
6. Midranges PCA (eq. 7): eigen-decomposition of Zr'Zr  ->  (lambda^r, u^r)
7. Global reconstruction (eq. 8-10): an orthogonal Procrustes rotation A
   (p x p) that best aligns Zr onto Zc; the rotated & midpoint-projected
   radii scores Psi_r = (Zr @ A) @ u^c, combined with the midpoint
   scores Psi_c = Zc @ u^c, give the interval principal-component
   coordinates:
       I[psi]_{i,alpha} = [ psi^c_{i,alpha} - psi_r_{i,alpha},
                             psi^c_{i,alpha} + psi_r_{i,alpha} ]
8. Explained-inertia decomposition (eq. 11): the total inertia p is
   split into a midpoints part, a radii part, and a "rotation"
   (connection) part, the latter from the singular values of Zc'Zr.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .interval_algebra import Interval, IntervalMatrix


@dataclass
class MidRadPCAResult:
    correlation_matrix: np.ndarray          # R2, p x p, trace == p
    midpoint_eigenvalues: np.ndarray         # lambda^c, decreasing
    midpoint_eigenvectors: np.ndarray        # u^c, columns
    radius_eigenvalues: np.ndarray           # lambda^r, decreasing
    radius_eigenvectors: np.ndarray          # u^r, columns
    rotation_singular_values: np.ndarray     # lambda^cr (diag of Lambda_cr)
    rotation_matrix: np.ndarray              # A = Q P' (p x p, orthogonal)
    midpoint_scores: np.ndarray              # Psi^c = Zc @ u^c   (N x m)
    radius_scores_rotated: np.ndarray        # Psi_r = (Zr @ A) @ u^c (N x m)
    interval_scores: IntervalMatrix          # I[psi] : psi^c -/+ Psi_r  (N x m)
    explained_inertia: np.ndarray            # per-axis, eq. (11)-style, length m
    variable_names: list = None
    unit_names: list = None


class MidpointRadiusPCA:
    """
    Midpoints-and-Radii Principal Component Analysis for interval data
    (Palumbo & Lauro), as an alternative to the correlation-matrix
    interval-optimisation approach of `ipca.IntervalPCA`.

    Parameters
    ----------
    n_components : int or None
        Number of axes to keep (default: all p).
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.result_: MidRadPCAResult | None = None

    def fit(self, X_lo, X_hi, variable_names=None, unit_names=None) -> "MidpointRadiusPCA":
        X_lo = np.asarray(X_lo, dtype=float)
        X_hi = np.asarray(X_hi, dtype=float)
        n, p = X_lo.shape
        m = self.n_components or p

        # 1. midpoints / radii (Sec. 2)
        Xc = (X_lo + X_hi) / 2.0
        Xr = (X_hi - X_lo) / 2.0

        # Centring convention: only the *location* (midpoints) is
        # mean-centred, exactly as ordinary PCA centres its variables.
        # The radius (an interval's *width*, i.e. its internal
        # variation) is left untouched by this shift of location --
        # sliding an interval along the real line changes where it is,
        # not how wide it is. (The paper's own eq. for interval
        # subtraction, applied literally to a *non-degenerate* mean
        # interval I[xbar] with its own radius xbar^r, would instead
        # inflate every unit's radius by xbar^r, which has no natural
        # interpretation as "centring"; we therefore adopt the standard
        # convention used in practice for this method.)
        Xc = Xc - Xc.mean(axis=0, keepdims=True)

        # 2. variance-covariance matrix (eq. 4)
        VX = (Xc.T @ Xc) / n + (Xr.T @ Xr) / n + (Xc.T @ Xr + Xr.T @ Xc) / n

        # 3. standardization (Sec. 3.1)
        sigma2 = np.diag(VX).copy()
        sigma2[sigma2 <= 0] = 1.0  # guard against degenerate (zero-variance) columns
        Sigma_inv = np.diag(1.0 / np.sqrt(sigma2))
        Zc = Xc @ Sigma_inv
        Zr = Xr @ Sigma_inv

        # 4. correlation matrix (eq. 5), trace == p.
        #
        #    NOTE ON A PAPER TYPO/OMISSION: eq. (5) as printed,
        #    R2 = Zc'Zc + Zr'Zr + Zc'Zr + Zr'Zc, has NO 1/N factor,
        #    but VX (eq. 4) -- of which R2 is the standardized version --
        #    *does* carry a 1/N factor. Taking eq. (5) completely
        #    literally makes every diagonal entry of R2 equal to N
        #    (verified algebraically: R2_jj = (a+b+2c)/sigma_j^2 with
        #    sigma_j^2 = (a+b+2c)/N, which cancels to exactly N for any
        #    data), i.e. trace(R2) = N*p, contradicting the paper's own
        #    explicit statement "tr(R2) = p" a few lines later. Dividing
        #    eq. (5) by N (consistent with eq. 4's own normalization)
        #    resolves this: it makes every diagonal entry exactly 1 and
        #    trace(R2) = p, matching the text. We therefore apply 1/N
        #    here, and to the midpoints/midranges eigenproblems (eq. 6-7)
        #    and the Procrustes SVD input (eq. 9) for consistency, so
        #    that lambda^c, lambda^r and lambda^cr are all on the same
        #    normalized scale (as Table 1 of the paper requires for its
        #    eigenvalues to be directly comparable and summable).
        ZcTZc = (Zc.T @ Zc) / n
        ZrTZr = (Zr.T @ Zr) / n
        ZcTZr = (Zc.T @ Zr) / n
        R2 = ZcTZc + ZrTZr + ZcTZr + ZcTZr.T
        R2 = 0.5 * (R2 + R2.T)  # numerical symmetrization

        # 5. midpoints PCA (eq. 6): eigen-decomposition of (1/N) Zc'Zc
        lam_c, Uc = _sorted_eigh(ZcTZc)

        # 6. midranges PCA (eq. 7): eigen-decomposition of (1/N) Zr'Zr
        lam_r, Ur = _sorted_eigh(ZrTZr)

        # 7. Procrustes rotation aligning Zr onto Zc (eq. 8-9):
        #    A = Q P' where  (1/N) Zc'Zr = P Lambda_cr Q'  (SVD)
        Pmat, s_cr, Qt = np.linalg.svd(ZcTZr)
        A = Qt.T @ Pmat.T

        # rotated radii, projected onto the *midpoints'* own PC axes so
        # that midpoint and radius coordinates live in the same
        # factorial plane (Sec. 3.4, "rotated radii coordinates are
        # represented on the midpoints PC's as supplementary points")
        Psi_c = Zc @ Uc[:, :m]
        Psi_r = (Zr @ A) @ Uc[:, :m]

        # 8. interval PC scores (eq. 10)
        interval_scores = IntervalMatrix(Psi_c - np.abs(Psi_r), Psi_c + np.abs(Psi_r))

        # explained inertia per axis (eq. 11-style). The paper divides
        # by the *fixed* total inertia p ("total inertia is equal to
        # p"), not by the sum of lambda^c+lambda^r+lambda^cr -- the
        # latter is a data-dependent quantity that need not equal p
        # (each per-axis piece is not individually bounded above by 1),
        # exactly as observed in the paper's own Table 1, where the
        # three eigenvalue sets sum to slightly more than p. We match
        # the paper's own normalization (divide by p) here.
        explained = np.zeros(m)
        for a in range(m):
            explained[a] = (lam_c[a] + lam_r[a] + s_cr[a]) / p

        self.result_ = MidRadPCAResult(
            correlation_matrix=R2,
            midpoint_eigenvalues=lam_c[:m],
            midpoint_eigenvectors=Uc[:, :m],
            radius_eigenvalues=lam_r[:m],
            radius_eigenvectors=Ur[:, :m],
            rotation_singular_values=s_cr[:m],
            rotation_matrix=A,
            midpoint_scores=Psi_c,
            radius_scores_rotated=Psi_r,
            interval_scores=interval_scores,
            explained_inertia=explained,
            variable_names=variable_names or [f"X{j+1}" for j in range(p)],
            unit_names=unit_names or [f"unit{i+1}" for i in range(n)],
        )
        return self

    # -- convenience accessors -----------------------------------------------
    @property
    def scores_(self) -> IntervalMatrix:
        return self.result_.interval_scores

    def summary(self) -> str:
        r = self.result_
        p = len(r.midpoint_eigenvalues)
        lines = ["Midpoint-Radius PCA summary", "=" * 40]
        lines.append("axis   lambda_c   lambda_r   lambda_cr   explained")
        for a in range(p):
            lines.append(
                f"  {a+1:2d}   {r.midpoint_eigenvalues[a]:8.4f}   "
                f"{r.radius_eigenvalues[a]:8.4f}   {r.rotation_singular_values[a]:9.4f}   "
                f"{r.explained_inertia[a]*100:6.2f}%"
            )
        if np.any(r.explained_inertia > 1.0):
            lines.append(
                "NOTE: a per-axis explained-inertia figure above 100% can occur "
                "-- lambda_c, lambda_r and lambda_cr come from three separate "
                "eigen-problems and are not individually bounded by the total "
                "inertia p (see midrad_pca.py's docstring and "
                "tests/test_midrad_pca_benchmark.py); it signals strong "
                "overlap between a unit's midpoint and radius variability, "
                "not a computation error."
            )
        return "\n".join(lines)


def _sorted_eigh(M: np.ndarray):
    """Eigen-decomposition of a symmetric matrix, sorted by decreasing
    eigenvalue (matching the paper's convention for ordering axes)."""
    w, V = np.linalg.eigh(M)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0.0, None)  # numerical guard: correlation-derived, PSD in theory
    V = V[:, order]
    return w, V
