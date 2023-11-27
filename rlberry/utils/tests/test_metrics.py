import pytest
import numpy as np
from rlberry.utils.metrics import metric_lp


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_metrics(dim):
    y = np.zeros(dim)
    x = np.ones(dim)
    scaling_1 = np.ones(dim)
    scaling_2 = 0.5 * np.ones(dim)

    for p in range(1, 10):
        assert np.abs(metric_lp(x, y, p, scaling_1) - np.power(dim, 1.0 / p)) < 1e-15
        assert (
                np.abs(metric_lp(x, y, p, scaling_2) - 2 * np.power(dim, 1.0 / p)) < 1e-15
        )


def test_metric_lp_zero_vectors():
    x = np.zeros(3)
    y = np.zeros(3)
    scaling = np.ones(3)
    for p in [1, 2, np.inf]:
        assert metric_lp(x, y, p, scaling) == 0.0


def test_metric_lp_negative_elements():
    x = np.array([-1, -2, -3])
    y = np.array([1, 2, 3])
    scaling = np.ones(3)
    for p in [1, 2, np.inf]:
        result = metric_lp(x, y, p, scaling)
        expected = np.linalg.norm((x - y) / scaling, ord=p)
        assert np.isclose(result, expected)


def test_metric_lp_infinity_norm():
    x = np.array([1, -2, 3])
    y = np.array([-1, 2, -3])
    scaling = np.ones(3)
    result = metric_lp(x, y, np.inf, scaling)
    expected = np.linalg.norm((x - y) / scaling, ord=np.inf)
    assert np.isclose(result, expected)


def test_metric_lp_different_scaling():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    scaling = np.array([1, 0.5, 0.25])
    for p in [1, 2, np.inf]:
        result = metric_lp(x, y, p, scaling)
        expected = np.linalg.norm((x - y) / scaling, ord=p)
        assert np.isclose(result, expected)


def test_metric_lp_input_validation():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    scaling = np.array([1, 0.5])
    with pytest.raises(AssertionError):
        metric_lp(x, y, 2, scaling)
