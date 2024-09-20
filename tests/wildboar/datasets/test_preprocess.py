import numpy as np
import pytest

from wildboar.datasets import load_ering, load_gun_point
from wildboar.datasets.preprocess import (
    Interpolate,
    MaxAbsScale,
    MinMaxScale,
    Standardize,
    Truncate,
)
from wildboar.utils.estimator_checks import check_estimator
from wildboar.utils.variable_len import EOS


@pytest.mark.parametrize(
    "estimator",
    [
        Truncate(),
        MinMaxScale(),
        MaxAbsScale(),
        Interpolate(),
        Standardize(),
    ],
)
def test_estimator_checks(estimator):
    check_estimator(estimator)


def test_truncate_2d():
    t = Truncate()
    a = np.arange(10 * 10, dtype=float).reshape(10, 10)
    a[0, 5] = EOS
    a[0, 1] = np.nan
    n = t.fit_transform(a)
    assert n.shape == (10, 5)


def test_truncate_3d():
    t = Truncate()
    a = np.arange(10 * 3 * 10, dtype=float).reshape(10, 3, 10)
    a[0, 0, 5] = EOS
    a[1, 2, 4] = EOS
    n = t.fit_transform(a)
    assert n.shape == (10, 3, 4)


@pytest.mark.parametrize("load_data", ["uts", "mts"], indirect=True)
def test_standardize(load_data):
    X, _ = load_data
    X += np.random.uniform(0, 12, np.prod(X.shape)).reshape(X.shape)
    X_s = Standardize().fit_transform(X)
    np.testing.assert_allclose(np.mean(X_s, axis=-1), 0.0, atol=0.000001)
    np.testing.assert_allclose(np.std(X_s, axis=-1), 1.0, atol=0.000001)


@pytest.mark.parametrize("load_data", ["uts", "mts"], indirect=["load_data"])
@pytest.mark.parametrize("min,max", [(0, 1), (1, 20), (-1, 1)])
def test_minmax_scale(load_data, min, max):
    X, _ = load_data
    X_s = MinMaxScale(min=min, max=max).fit_transform(X)
    assert np.min(X_s) == min
    assert np.max(X_s) == max


@pytest.mark.parametrize("load_data", ["uts", "mts"], indirect=["load_data"])
def test_maxabs_scale(load_data):
    X, _ = load_data
    X_s = MaxAbsScale().fit_transform(X)
    if X.ndim == 2:
        assert X_s[0, 0] == X[0, 0] / np.nanmax(X[0])
    else:
        assert X_s[0, 0, 0] == X[0, 0, 0] / np.nanmax(X[0, 0])


@pytest.mark.parametrize(
    "method,desired",
    [
        ("linear", [-1.184943, -1.157614, -1.130285, -1.102956, -1.075627, -1.048298]),
        ("cubic", [-1.212705, -1.191666, -1.156486, -1.114495, -1.073025, -1.039406]),
        ("pchip", [-1.201675, -1.174346, -1.136978, -1.096263, -1.058895, -1.031566]),
    ],
)
def test_interpolate_3d(ering, method, desired):
    X = ering[0].copy()
    X[0, 0, 4:10] = np.nan
    X_s = Interpolate(method=method).fit_transform(X)
    np.testing.assert_allclose(
        X_s[0, 0, 4:10],
        desired,
        atol=0.0001,
    )


@pytest.mark.parametrize(
    "method,desired",
    [
        ("linear", [-0.813844, -0.813468, -0.813091, -0.812715, -0.812339, -0.811963]),
        ("cubic", [-0.812724, -0.812383, -0.812728, -0.81329, -0.8136, -0.813188]),
        ("pchip", [-0.813641, -0.813296, -0.813086, -0.81291, -0.812668, -0.81226]),
    ],
)
def test_interpolate_2d(gun_point, method, desired):
    X = gun_point[0].copy()
    X[10, 4:10] = np.nan
    X_s = Interpolate(method=method).fit_transform(X)
    np.testing.assert_allclose(
        X_s[10, 4:10],
        desired,
        atol=0.0001,
    )


@pytest.mark.parametrize(
    "estimator",
    [
        Interpolate(method="linear"),
        Interpolate(method="cubic"),
        Interpolate(method="pchip"),
    ],
)
def test_interpolate_2d_benchmark(estimator, benchmark, X):
    benchmark(estimator.fit, X=X)
