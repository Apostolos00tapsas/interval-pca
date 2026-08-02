"""
datasets.py
-----------
The "Oil data" benchmark (Ichino 1988), reproduced exactly from Table 2
of Gioia & Lauro (2006), Section 5 ("Numerical results").  Eight oils,
four interval-valued variables: Specific gravity, Freezing point,
Iodine value, Saponification.

This is the same dataset the paper itself uses to illustrate IPCA, so
it is the natural benchmark for validating this implementation's
correlation matrix, eigenvalues, and explained-variance intervals
against the numbers actually published in the paper.
"""

import numpy as np

OIL_UNITS = ["Linseed", "Perilla", "Cotton", "Sesame",
             "Camellia", "Olive", "Beef", "Hog"]

OIL_VARIABLES = ["Spec.gravity", "Freezing point", "Iodine value", "Saponification"]

# columns: Spec.gravity, Freezing point, Iodine value, Saponification
# each entry is (lo, hi), copied verbatim from Table 2 of the paper.
_OIL_RAW = {
    "Linseed":  [(0.93, 0.94), (-27, -18), (170, 204), (118, 196)],
    "Perilla":  [(0.93, 0.94), (-5, -4),   (192, 208), (188, 197)],
    "Cotton":   [(0.92, 0.92), (-6, -1),   (99, 113),  (189, 198)],
    "Sesame":   [(0.92, 0.93), (-6, -4),   (104, 116), (187, 193)],
    "Camellia": [(0.92, 0.92), (-21, -15), (80, 82),   (189, 193)],
    "Olive":    [(0.91, 0.92), (0, 6),     (79, 90),   (187, 196)],
    "Beef":     [(0.86, 0.87), (30, 38),   (40, 48),   (190, 199)],
    "Hog":      [(0.86, 0.86), (22, 32),   (53, 77),   (190, 202)],
}


def load_oil_dataset():
    """
    Returns (X_lo, X_hi, unit_names, variable_names) for the Oil
    dataset of Table 2 in Gioia & Lauro (2006), transcribed verbatim.

    DATA QUALITY NOTE: Linseed's Saponification interval is [118, 196]
    in this table, a radius of 39 around a centre of 157. Every other
    oil's Saponification interval sits inside [187, 202] (radius <= 8).
    Cross-validating against Palumbo & Lauro (2003)'s midpoints-radii
    PCA analysis of the *same* Ichino oils dataset (see
    midrad_pca.py / tests/test_midrad_pca_benchmark.py) strongly
    suggests the lower bound "118" is a scanning/transcription artefact
    for "188" (a single-digit OCR error, and the value that would fit
    the pattern of every other unit): substituting 188 brings this
    library's midpoints-eigenvalue reproduction of Palumbo & Lauro's
    Table 1 into close numerical agreement, whereas the literal "118"
    does not. We deliberately keep the dataset *exactly as printed* in
    Table 2 by default (least-surprise, and easy to audit against the
    source), and expose the corrected variant separately via
    `load_oil_dataset_corrected()` for anyone who wants to reproduce
    the closer match or investigate the sensitivity themselves.
    """
    n = len(OIL_UNITS)
    p = len(OIL_VARIABLES)
    X_lo = np.zeros((n, p))
    X_hi = np.zeros((n, p))
    for i, unit in enumerate(OIL_UNITS):
        for j in range(p):
            lo, hi = _OIL_RAW[unit][j]
            X_lo[i, j] = lo
            X_hi[i, j] = hi
    return X_lo, X_hi, list(OIL_UNITS), list(OIL_VARIABLES)


def load_oil_dataset_corrected():
    """
    Same as load_oil_dataset(), except Linseed's Saponification lower
    bound is changed from 118 to 188 -- see the data-quality note in
    load_oil_dataset()'s docstring. This is a documented hypothesis
    about a likely transcription artefact, not a claim about the
    "true" original data; use load_oil_dataset() for anything that
    should match Table 2 exactly.
    """
    X_lo, X_hi, units, variables = load_oil_dataset()
    X_lo = X_lo.copy()
    X_lo[units.index("Linseed"), variables.index("Saponification")] = 188.0
    return X_lo, X_hi, units, variables


# --------------------------------------------------------------------------
# Values *published in the paper* (Table 2 correlation matrix, and the
# eigenvalues / explained variance reported in Section 5), used as the
# ground truth for the validity check in verify.py
# --------------------------------------------------------------------------

PAPER_CORRELATION_INTERVALS = {
    ("Spec.gravity", "Freezing point"): (-0.97, -0.80),
    ("Spec.gravity", "Iodine value"): (0.62, 0.88),
    ("Spec.gravity", "Saponification"): (-0.64, -0.16),
    ("Freezing point", "Iodine value"): (-0.77, -0.52),
    ("Freezing point", "Saponification"): (0.30, 0.75),
    ("Iodine value", "Saponification"): (-0.77, -0.34),
}

PAPER_EIGENVALUES = [
    (2.45, 3.40),
    (0.68, 1.11),
    (0.22, 0.33),
    (0.00, 0.08),
]

PAPER_EXPLAINED_VARIANCE_PCT = [
    (61, 86),
    (15, 32),
    (4, 9),
    (0, 2),
]

# --------------------------------------------------------------------------
# Palumbo & Lauro (2003), "A PCA for interval-valued data based on
# midpoints and radii" -- Table 1, same Ichino oils dataset. Used as
# ground truth for tests/test_midrad_pca_benchmark.py.
# --------------------------------------------------------------------------

PALUMBO_LAURO_TABLE1_MIDPOINTS = [2.359, 0.332, 0.182, 0.031]
PALUMBO_LAURO_TABLE1_MIDRANGES = [0.252, 0.008, 0.003, 0.002]
PALUMBO_LAURO_TABLE1_ROTATION = [0.634, 0.280, 0.033, 0.008]
