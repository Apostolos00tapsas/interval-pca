"""
interval_algebra.py
--------------------
Basic interval algebra as defined in:

    Gioia, F. & Lauro, C.N. (2006), "Principal Component Analysis on
    Interval Data", Computational Statistics 21, 343-363.
    (Section 2: "Definitions notations and basic facts")

and the classical reference:

    Moore, R.E. (1966), "Interval Analysis", Prentice Hall.

An interval [a, b] with a <= b is the set {x : a <= x <= b}.  For any
elementary operator o in {+, -, *, /} the interval result is defined as

    [a, b] o [c, d] = { x o y | x in [a, b], y in [c, d] }            (2.1.1)

which reduces, for real numbers, to the closed-form rules in (2.1.2):

    [a,b] + [c,d] = [a+c, b+d]
    [a,b] - [c,d] = [a-d, b-c]
    [a,b] * [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
    [a,b] / [c,d] = [a,b] * [1/d, 1/c]      (undefined if 0 in [c,d])

This module implements:
  * Interval        -- a single scalar interval, operator overloaded
  * IntervalVector   -- an array of intervals (thin numpy wrapper)
  * IntervalMatrix   -- an interval matrix, with the interval matrix
                        product of Definition 2.2.3
"""

from __future__ import annotations
import numpy as np
from numbers import Real


# --------------------------------------------------------------------------
# Scalar interval
# --------------------------------------------------------------------------

class Interval:
    """A single real interval [lo, hi], lo <= hi (degenerate if lo == hi)."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        if hi is None:
            hi = lo
        lo = float(lo)
        hi = float(hi)
        if lo > hi:
            raise ValueError(f"Invalid interval: lo={lo} > hi={hi}")
        self.lo = lo
        self.hi = hi

    # -- construction helpers -------------------------------------------------
    @staticmethod
    def _as_interval(other) -> "Interval":
        if isinstance(other, Interval):
            return other
        if isinstance(other, Real):
            return Interval(other, other)
        raise TypeError(f"Cannot interpret {other!r} as an Interval")

    @classmethod
    def degenerate(cls, x: float) -> "Interval":
        """A 'thin' interval [x, x], equivalent to the real number x."""
        return cls(x, x)

    # -- basic properties -------------------------------------------------
    @property
    def mid(self) -> float:
        """midpoint / centre, X_c = (X_lo + X_hi) / 2 (Def. 2.2.1)."""
        return (self.lo + self.hi) / 2.0

    @property
    def rad(self) -> float:
        """radius, DeltaX = (X_hi - X_lo) / 2 (Def. 2.2.1)."""
        return (self.hi - self.lo) / 2.0

    @property
    def width(self) -> float:
        """diameter d([x]) = hi - lo."""
        return self.hi - self.lo

    def is_degenerate(self, tol: float = 0.0) -> bool:
        return self.width <= tol

    def contains(self, other) -> bool:
        """Set inclusion: self supseteq other (Theorem 2.1.1 ordering)."""
        other = Interval._as_interval(other)
        return self.lo <= other.lo and other.hi <= self.hi

    def __contains__(self, x) -> bool:
        if isinstance(x, Interval):
            return self.contains(x)
        return self.lo <= x <= self.hi

    # -- arithmetic (2.1.2) -------------------------------------------------
    def __add__(self, other) -> "Interval":
        o = Interval._as_interval(other)
        return Interval(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other) -> "Interval":
        o = Interval._as_interval(other)
        return Interval(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, other) -> "Interval":
        return Interval._as_interval(other).__sub__(self)

    def __mul__(self, other) -> "Interval":
        o = Interval._as_interval(other)
        products = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Interval(min(products), max(products))

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Interval":
        o = Interval._as_interval(other)
        if o.lo <= 0.0 <= o.hi:
            raise ZeroDivisionError(
                f"0 is contained in the divisor interval [{o.lo}, {o.hi}]; "
                "interval division is undefined (Sec. 2.1)."
            )
        inv = Interval(1.0 / o.hi, 1.0 / o.lo)
        return self * inv

    def __rtruediv__(self, other) -> "Interval":
        return Interval._as_interval(other).__truediv__(self)

    # -- set operations -------------------------------------------------
    def union_hull(self, other) -> "Interval":
        """Interval hull of the union (smallest interval containing both)."""
        o = Interval._as_interval(other)
        return Interval(min(self.lo, o.lo), max(self.hi, o.hi))

    def intersection(self, other) -> "Interval | None":
        o = Interval._as_interval(other)
        lo = max(self.lo, o.lo)
        hi = min(self.hi, o.hi)
        if lo > hi:
            return None
        return Interval(lo, hi)

    # -- comparisons / misc -------------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi

    def __repr__(self) -> str:
        return f"[{self.lo:.6g}, {self.hi:.6g}]"

    def __hash__(self):
        return hash((self.lo, self.hi))


# --------------------------------------------------------------------------
# Vectorised interval containers (numpy-backed: two float arrays, lo & hi)
# --------------------------------------------------------------------------

def _check_lo_hi(lo: np.ndarray, hi: np.ndarray):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.shape != hi.shape:
        raise ValueError("lo and hi must have the same shape")
    if np.any(lo > hi + 1e-12):
        raise ValueError("Every entry must satisfy lo <= hi")
    return lo, hi


class IntervalArray:
    """
    A numpy-backed array of intervals (vector or matrix), stored as two
    parallel arrays `lo` and `hi`.  This underlies IntervalVector and
    IntervalMatrix and implements elementwise arithmetic (Def. 2.2.3,
    first bullet) exactly as in (2.1.2), applied elementwise.
    """

    def __init__(self, lo, hi=None):
        if hi is None and isinstance(lo, IntervalArray):
            hi = lo.hi
            lo = lo.lo
        elif hi is None:
            lo, hi = np.asarray(lo, dtype=float), np.asarray(lo, dtype=float)
        self.lo, self.hi = _check_lo_hi(lo, hi)

    @property
    def shape(self):
        return self.lo.shape

    @property
    def mid(self) -> np.ndarray:
        """Centre matrix X_c = (X_lo + X_hi)/2 (Def. 2.2.1)."""
        return (self.lo + self.hi) / 2.0

    @property
    def rad(self) -> np.ndarray:
        """Radius matrix Delta_X = (X_hi - X_lo)/2 (Def. 2.2.1)."""
        return (self.hi - self.lo) / 2.0

    @property
    def width(self) -> np.ndarray:
        return self.hi - self.lo

    @classmethod
    def from_intervals(cls, arr) -> "IntervalArray":
        """Build from a nested list/array of Interval objects."""
        arr = np.asarray(arr, dtype=object)
        lo = np.vectorize(lambda iv: iv.lo)(arr).astype(float)
        hi = np.vectorize(lambda iv: iv.hi)(arr).astype(float)
        return cls(lo, hi)

    @classmethod
    def degenerate(cls, values) -> "IntervalArray":
        v = np.asarray(values, dtype=float)
        return cls(v.copy(), v.copy())

    def to_object_array(self) -> np.ndarray:
        """Return an array of scalar Interval objects (for pretty printing)."""
        out = np.empty(self.shape, dtype=object)
        it = np.nditer(self.lo, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            out[idx] = Interval(self.lo[idx], self.hi[idx])
        return out

    # -- elementwise arithmetic, per (2.1.2) applied elementwise -----------
    def __add__(self, other):
        other = self._coerce(other)
        return IntervalArray(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return IntervalArray(-self.hi, -self.lo)

    def __sub__(self, other):
        other = self._coerce(other)
        return IntervalArray(self.lo - other.hi, self.hi - other.lo)

    def __rsub__(self, other):
        other = self._coerce(other)
        return other.__sub__(self)

    def __mul__(self, other):
        """Elementwise (Hadamard) interval product."""
        other = self._coerce(other)
        p1 = self.lo * other.lo
        p2 = self.lo * other.hi
        p3 = self.hi * other.lo
        p4 = self.hi * other.hi
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        return IntervalArray(lo, hi)

    __rmul__ = __mul__

    def _coerce(self, other) -> "IntervalArray":
        if isinstance(other, IntervalArray):
            return other
        if isinstance(other, Interval):
            return IntervalArray(np.full(self.shape, other.lo),
                                  np.full(self.shape, other.hi))
        arr = np.asarray(other, dtype=float)
        return IntervalArray(arr, arr)

    def transpose(self):
        return IntervalArray(self.lo.T, self.hi.T)

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, key):
        return IntervalArray(self.lo[key], self.hi[key])

    def __repr__(self):
        return f"IntervalArray(shape={self.shape})\n lo=\n{self.lo}\n hi=\n{self.hi}"


class IntervalVector(IntervalArray):
    """1-D IntervalArray."""
    pass


class IntervalMatrix(IntervalArray):
    """
    2-D IntervalArray implementing the interval matrix product of
    Definition 2.2.3:

        (X^I Y^I)_ij = sum_v X_iv Y_vj    (interval sum of interval products)

    computed exactly (not approximated) via the corner-product rule
    (2.1.2) applied to every term before summing.
    """

    def matmul(self, other: "IntervalMatrix") -> "IntervalMatrix":
        other = self._coerce_matrix(other)
        A_lo, A_hi = self.lo, self.hi          # (n, r)
        B_lo, B_hi = other.lo, other.hi        # (r, p)
        # broadcast to (n, r, p) for the four corner products
        P1 = A_lo[:, :, None] * B_lo[None, :, :]
        P2 = A_lo[:, :, None] * B_hi[None, :, :]
        P3 = A_hi[:, :, None] * B_lo[None, :, :]
        P4 = A_hi[:, :, None] * B_hi[None, :, :]
        lo_terms = np.minimum(np.minimum(P1, P2), np.minimum(P3, P4))
        hi_terms = np.maximum(np.maximum(P1, P2), np.maximum(P3, P4))
        C_lo = lo_terms.sum(axis=1)
        C_hi = hi_terms.sum(axis=1)
        return IntervalMatrix(C_lo, C_hi)

    def __matmul__(self, other):
        return self.matmul(other)

    @staticmethod
    def _coerce_matrix(other) -> "IntervalMatrix":
        if isinstance(other, IntervalMatrix):
            return other
        if isinstance(other, IntervalArray):
            return IntervalMatrix(other.lo, other.hi)
        arr = np.asarray(other, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return IntervalMatrix(arr, arr)

    def scalar_multiply(self, k: Interval) -> "IntervalMatrix":
        """K * X^I = X^I * K for a constant interval K (Def. 2.2.3)."""
        vals = (self.lo * k.lo, self.lo * k.hi, self.hi * k.lo, self.hi * k.hi)
        lo = np.minimum.reduce(vals)
        hi = np.maximum.reduce(vals)
        return IntervalMatrix(lo, hi)

    def is_symmetric(self, tol: float = 1e-9) -> bool:
        """
        An n x n interval matrix X^I is symmetric (Def. 2.2.2) iff both
        the lower-bound matrix X and the upper-bound matrix X-bar are
        symmetric.
        """
        return (np.allclose(self.lo, self.lo.T, atol=tol) and
                np.allclose(self.hi, self.hi.T, atol=tol))

    def symmetrize(self) -> "IntervalMatrix":
        """
        Force symmetry via Def. 2.2.2:
            X^I_s = [ (1/2)(X + X^T), (1/2)(Xbar + Xbar^T) ]
        """
        lo = 0.5 * (self.lo + self.lo.T)
        hi = 0.5 * (self.hi + self.hi.T)
        return IntervalMatrix(lo, hi)
