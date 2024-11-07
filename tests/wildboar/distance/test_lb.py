import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from wildboar.datasets import load_gun_point
from wildboar.distance.lb import (
    DtwKeoghLowerBound,
    DtwKimLowerBound,
    PaaLowerBound,
    SaxLowerBound,
)
from wildboar.utils._testing import (
    assert_exhaustive_parameter_checks,
    assert_parameter_checks,
)
from wildboar.utils.estimator_checks import check_estimator


@pytest.mark.parametrize(
    "estimator",
    [
        DtwKeoghLowerBound(),
        DtwKimLowerBound(),
        PaaLowerBound(),
        SaxLowerBound(),
    ],
)
def test_estimator_checks(estimator):
    check_estimator(estimator)
    assert_parameter_checks(estimator)
    assert_exhaustive_parameter_checks(estimator)


@pytest.mark.parametrize(
    "r,expected",
    [
        (
            0.1,
            np.array(
                [
                    [3.30092013, 2.61546763, 1.75774304, 1.98565672, 3.05331929],
                    [3.59510336, 3.58114675, 2.67428025, 3.00632452, 4.53328931],
                    [4.19370778, 5.18038046, 3.78883294, 4.8289548, 7.57916512],
                ]
            ),
        ),
        (
            0.5,
            np.array(
                [
                    [1.8704832, 2.15211288, 1.6825272, 1.63988294, 2.68757666],
                    [3.04895091, 3.27976414, 2.48781032, 2.84358674, 3.7759067],
                    [3.72037408, 3.92365404, 3.20265619, 3.52869195, 4.39514919],
                ]
            ),
        ),
        (
            1.0,
            np.array(
                [
                    [1.8704832, 2.15211288, 1.45662298, 1.63988294, 2.68757666],
                    [3.04895091, 3.27976414, 2.48781032, 2.84358674, 3.7759067],
                    [3.72037408, 3.92365404, 3.20265619, 3.52869195, 4.39514919],
                ]
            ),
        ),
    ],
)
def test_dtw_keogh_lower_bound(r, expected):
    X, y = load_gun_point()
    Y = X[:5]
    X = X[10:13]

    actual = DtwKeoghLowerBound(r=r).fit(Y).transform(X)
    assert_almost_equal(actual, expected)


def test_dtw_kim_lower_bound():
    X, y = load_gun_point()
    Y = X[:5]
    X = X[10:13]
    expected = np.array(
        [
            [0.40783531, 0.42069802, 0.21365168, 0.19725256, 0.46663072],
            [1.37084334, 1.38671857, 0.86028071, 0.88235554, 1.52141843],
            [1.13174997, 1.13667356, 0.65664954, 0.68616908, 1.26697248],
        ]
    )
    actual = DtwKimLowerBound().fit(Y).transform(X)

    assert_almost_equal(actual, expected)


@pytest.mark.parametrize("lower_bound", [DtwKeoghLowerBound(), DtwKimLowerBound()])
@pytest.mark.parametrize("r", np.linspace(0, 1, 10, endpoint=True))
def test_dtw_lower_bound_lower_bounds_dtw(lower_bound, r):
    from wildboar.distance import pairwise_distance

    X, _ = load_gun_point()
    X = X[:30]
    distance = pairwise_distance(X, metric="dtw", metric_params={"r": r})
    if hasattr(lower_bound, "r"):
        lower_bound.set_params(r=r)
    lower_bound = lower_bound.fit_transform(X)
    assert (lower_bound <= distance).all()


@pytest.mark.parametrize("n_bins", [2, 10, 20, 50, 100])
@pytest.mark.parametrize("n_intervals", [10, 12, 18, 22])
def test_sax_lower_bounds_euclidean(n_bins, n_intervals):
    from wildboar.datasets.preprocess import standardize
    from wildboar.distance import pairwise_distance

    X, _ = load_gun_point()
    X = standardize(X[:30])
    distance = pairwise_distance(X, metric="euclidean")
    lower_bound = SaxLowerBound(n_intervals=n_intervals, n_bins=n_bins).fit_transform(X)
    assert (lower_bound <= distance).all()


@pytest.mark.parametrize("n_intervals", [10, 12, 18, 22])
def test_paa_lower_bounds_euclidean(n_intervals):
    from wildboar.datasets.preprocess import standardize
    from wildboar.distance import pairwise_distance

    X, _ = load_gun_point()
    X = standardize(X[:30])
    distance = pairwise_distance(X, metric="euclidean")
    lower_bound = PaaLowerBound(n_intervals=n_intervals).fit_transform(X)
    assert (lower_bound <= distance).all()
