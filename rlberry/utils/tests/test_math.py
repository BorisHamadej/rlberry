import numpy as np
import pytest
from rlberry.utils.math import lmap


def test_lmap_basic_functionality():
    v = np.array([0.5])
    x = (0, 1)
    y = (0, 10)
    expected = np.array([5.0])
    np.testing.assert_array_almost_equal(lmap(v, x, y), expected)


def test_lmap_reverse_intervals():
    v = np.array([5])
    x = (0, 10)
    y = (0, 1)
    expected = np.array([0.5])
    np.testing.assert_array_almost_equal(lmap(v, x, y), expected)


def test_lmap_vectorized_input():
    v = np.array([0.25, 0.5, 0.75])
    x = (0, 1)
    y = (0, 10)
    expected = np.array([2.5, 5.0, 7.5])
    np.testing.assert_array_almost_equal(lmap(v, x, y), expected)


def test_lmap_edge_cases():
    v = np.array([0, 1])
    x = (0, 1)
    y = (0, 10)
    expected = np.array([0, 10])
    np.testing.assert_array_almost_equal(lmap(v, x, y), expected)


def test_lmap_exception_handling():
    v = np.array([0.5])
    x = (0, 0)
    y = (0, 10)
    with pytest.raises(ZeroDivisionError):
        lmap(v, x, y)
