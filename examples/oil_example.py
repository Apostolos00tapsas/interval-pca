"""
oil_example.py
---------------
Minimal usage example: load the Oil dataset (Table 2, Gioia & Lauro
2006), run Interval PCA, and print the results.

Run with:   python examples/oil_example.py
"""

import sys
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from interval_pca.datasets import load_oil_dataset
from interval_pca.ipca import IntervalPCA
from interval_pca.interval_algebra import Interval
from interval_pca.vis import plot_pca_rectangles

if __name__ == "__main__":
    X_lo, X_hi, units, variables = load_oil_dataset()

    print("#" * 70)
    print("# 1. Gioia & Lauro (2006) IntervalPCA")
    print("#" * 70)
    pca = IntervalPCA(standardize=True, n_restarts=8, seed=0)
    pca.fit(X_lo, X_hi, variable_names=variables, unit_names=units)

    print(pca.summary())

    print("\nInterval principal component scores (axis 1, axis 2):")
    scores = pca.scores_
    for i, unit in enumerate(units):
        pc1 = Interval(scores.lo[i, 0], scores.hi[i, 0])
        pc2 = Interval(scores.lo[i, 1], scores.hi[i, 1])
        print(f"  {unit:10s} PC1={pc1}  PC2={pc2}")

    print("\nInterval correlations of each variable with axis 1:")
    r = pca.result_
    for j, var in enumerate(variables):
        iv = Interval(r.var_axis_correlations.lo[j, 0], r.var_axis_correlations.hi[j, 0])
        print(f"  {var:15s} {iv}")

    print("\nAbsolute contribution of each unit to axis 1 (eq. 4.3.2):")
    for i, unit in enumerate(units):
        c = pca.absolute_contribution(i, axis=0)
        print(f"  {unit:10s} {c}")

    # ------------------------------------------------------------------
    # A tiny illustration of the raw interval-algebra layer on its own
    # ------------------------------------------------------------------
    from interval_pca.interval_algebra import Interval as I
    print("\nBasic interval algebra (Section 2.1):")
    a, b = I(1, 2), I(3, 5)
    print(f"  {a} + {b} = {a + b}")
    print(f"  {a} - {b} = {a - b}")
    print(f"  {a} * {b} = {a * b}")
    print(f"  {a} / {b} = {a / b}")

    # ------------------------------------------------------------------
    # Alternative method: Palumbo & Lauro's midpoints-and-radii PCA,
    # which avoids the nonlinear-interval-optimisation oversizing seen
    # above for pairs involving Saponification (see the README and
    # tests/test_oil_benchmark.py vs tests/test_midrad_pca_benchmark.py)
    # ------------------------------------------------------------------
    from interval_pca.midrad_pca import MidpointRadiusPCA

    print("\n" + "#" * 70)
    print("# 2. Palumbo & Lauro (2003) MidpointRadiusPCA")
    print("#" * 70)
    mr = MidpointRadiusPCA()
    mr.fit(X_lo, X_hi, variable_names=variables, unit_names=units)
    print(mr.summary())

    print("\nInterval PC1 scores (midpoints-and-radii method):")
    mr_scores = mr.scores_
    for i, unit in enumerate(units):
        pc1 = Interval(mr_scores.lo[i, 0], mr_scores.hi[i, 0])
        print(f"  {unit:10s} PC1={pc1}")

    # Calculate scores
    print("Plot with 1 line!")
    fig, ax = plot_pca_rectangles(mr_scores.lo, mr_scores.hi, labels=units, title="Oil Dataset - Midpoint & Radius PCA")
    plt.show()