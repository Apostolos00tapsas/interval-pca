"""
Tests for interval_pca.interval_algebra, checking the arithmetic rules
against eq. (2.1.2) of Gioia & Lauro (2006) and basic algebraic
properties stated in Section 2.1 of the paper (commutativity,
associativity, sub-distributivity, degenerate-interval = real number).
"""
import numpy as np
import pytest

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from interval_pca.interval_algebra import Interval, IntervalMatrix


def test_addition_matches_2_1_2():
    a = Interval(1, 2)
    b = Interval(3, 5)
    c = a + b
    assert (c.lo, c.hi) == (4, 7)


def test_subtraction_matches_2_1_2():
    a = Interval(1, 2)
    b = Interval(3, 5)
    c = a - b
    # [a,b] - [c,d] = [a-d, b-c]
    assert (c.lo, c.hi) == (1 - 5, 2 - 3)


def test_multiplication_matches_2_1_2():
    a = Interval(-2, 3)
    b = Interval(-4, 1)
    c = a * b
    corners = [(-2) * (-4), (-2) * 1, 3 * (-4), 3 * 1]
    assert c.lo == min(corners)
    assert c.hi == max(corners)


def test_division_undefined_when_zero_in_divisor():
    a = Interval(1, 2)
    b = Interval(-1, 1)
    with pytest.raises(ZeroDivisionError):
        _ = a / b


def test_division_matches_2_1_2():
    a = Interval(4, 8)
    b = Interval(2, 4)
    c = a / b
    # a/b = a * [1/4, 1/2]
    corners = [4 * 0.25, 4 * 0.5, 8 * 0.25, 8 * 0.5]
    assert c.lo == pytest.approx(min(corners))
    assert c.hi == pytest.approx(max(corners))


def test_degenerate_interval_behaves_like_real_number():
    a = Interval.degenerate(3.0)
    b = Interval.degenerate(4.0)
    assert (a + b) == Interval.degenerate(7.0)
    assert (a * b) == Interval.degenerate(12.0)


def test_commutativity_and_associativity_of_addition():
    a, b, c = Interval(1, 2), Interval(-3, 0), Interval(5, 6)
    assert (a + b) == (b + a)
    assert ((a + b) + c) == (a + (b + c))


def test_commutativity_of_multiplication():
    a, b = Interval(-2, 3), Interval(1, 4)
    assert (a * b) == (b * a)


def test_subdistributive_law_holds_as_inclusion():
    # x(y+z) subseteq xy + xz  (Sec. 2.1)
    x, y, z = Interval(1, 2), Interval(-1, 3), Interval(2, 5)
    left = x * (y + z)
    right = (x * y) + (x * z)
    assert right.contains(left)


def test_zero_and_one_are_units():
    a = Interval(2, 5)
    zero = Interval.degenerate(0.0)
    one = Interval.degenerate(1.0)
    assert (a + zero) == a
    assert (a * one) == a


def test_interval_hull_and_intersection():
    a = Interval(1, 3)
    b = Interval(2, 5)
    hull = a.union_hull(b)
    assert (hull.lo, hull.hi) == (1, 5)
    inter = a.intersection(b)
    assert (inter.lo, inter.hi) == (2, 3)
    c = Interval(10, 12)
    assert a.intersection(c) is None


def test_interval_matrix_product_matches_plain_matmul_for_degenerate_intervals():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 3))
    B = rng.normal(size=(3, 5))
    AI = IntervalMatrix(A, A)
    BI = IntervalMatrix(B, B)
    C = AI.matmul(BI)
    expected = A @ B
    assert np.allclose(C.lo, expected)
    assert np.allclose(C.hi, expected)


def test_interval_matrix_product_definition_2_2_3_small_case():
    # A = [[ [1,2] ]] (1x1), B = [[ [-1,1] ]] (1x1)
    # (A*B) should be interval product [1,2]*[-1,1] = [-2,2]
    AI = IntervalMatrix(np.array([[1.0]]), np.array([[2.0]]))
    BI = IntervalMatrix(np.array([[-1.0]]), np.array([[1.0]]))
    C = AI.matmul(BI)
    assert (C.lo[0, 0], C.hi[0, 0]) == (-2.0, 2.0)


def test_symmetric_interval_matrix_definition_2_2_2():
    lo = np.array([[1.0, 2.0], [1.5, 3.0]])
    hi = np.array([[1.0, 2.5], [2.0, 3.0]])
    M = IntervalMatrix(lo, hi)
    assert not M.is_symmetric()
    Ms = M.symmetrize()
    assert Ms.is_symmetric()


def test_inclusion_isotony():
    # if [a] subset [b] and [c] subset [d] then [a] o [c] subset [b] o [d]
    a, b = Interval(1, 2), Interval(0, 3)
    c, d = Interval(4, 5), Interval(3, 6)
    assert b.contains(a) and d.contains(c)
    for op in ("__add__", "__sub__", "__mul__"):
        left = getattr(a, op)(c)
        right = getattr(b, op)(d)
        assert right.contains(left), f"inclusion isotony failed for {op}"
