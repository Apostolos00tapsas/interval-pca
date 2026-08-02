"""
interval_pca
============
A small Python library implementing:

  1. Interval algebra (Moore-style interval arithmetic and interval
     matrix operations), per Section 2 of Gioia & Lauro (2006).
  2. Interval Principal Component Analysis (IPCA), per Sections 3-4 of
     the same paper, together with a benchmark-based validity check
     against the "Oil data" numerical example of Section 5.

References
----------
Gioia, F. & Lauro, C.N. (2006). "Principal Component Analysis on
Interval Data." Computational Statistics 21, 343-363.
"""

from .interval_algebra import Interval, IntervalArray, IntervalVector, IntervalMatrix
from .interval_stats import (
    interval_standardize,
    interval_standardize_column,
    interval_correlation_matrix,
    interval_correlation_pair,
)
from .ipca import (
    IntervalPCA,
    IPCAResult,
    interval_eigen_symmetric,
    interval_explained_variance,
    interval_absolute_contribution,
)
from .midrad_pca import MidpointRadiusPCA, MidRadPCAResult

__all__ = [
    "Interval", "IntervalArray", "IntervalVector", "IntervalMatrix",
    "interval_standardize", "interval_standardize_column",
    "interval_correlation_matrix", "interval_correlation_pair",
    "IntervalPCA", "IPCAResult",
    "interval_eigen_symmetric", "interval_explained_variance",
    "interval_absolute_contribution",
    "MidpointRadiusPCA", "MidRadPCAResult",
]

__version__ = "0.1.0"
