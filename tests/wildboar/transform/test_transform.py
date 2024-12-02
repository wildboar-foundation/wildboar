# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.utils.estimator_checks import check_estimators_pickle

from wildboar.datasets import load_gun_point
from wildboar.transform import (
    PAA,
    SAX,
    CastorTransform,
    DerivativeTransform,
    DilatedShapeletTransform,
    HydraTransform,
    IntervalTransform,
    MatrixProfileTransform,
    PivotTransform,
    QuantTransform,
    RandomShapeletTransform,
    RocketTransform,
    ShapeletTransform,
)
from wildboar.transform._diff import FftTransform
from wildboar.utils._testing import (
    assert_exhaustive_parameter_checks,
    assert_parameter_checks,
)
from wildboar.utils.estimator_checks import check_estimator


@pytest.mark.parametrize(
    "estimator, skip",
    [
        (IntervalTransform(), []),
        (IntervalTransform(intervals="dyadic", summarizer="quant"), []),
        (QuantTransform(), []),
        (RandomShapeletTransform(), []),
        (ShapeletTransform(), []),
        (RocketTransform(), ["max_size"]),
        (MatrixProfileTransform(), []),
        (PivotTransform(), ["metric_factories"]),
        (SAX(), []),
        (PAA(), []),
        (FftTransform(), []),
        (HydraTransform(), []),
        (DilatedShapeletTransform(), []),
        (CastorTransform(), []),
    ],
)
def test_estimator_checks(estimator, skip):
    check_estimator(estimator)
    assert_exhaustive_parameter_checks(estimator)
    assert_parameter_checks(estimator, skip=skip)


def test_random_shapelet_transform_equals_shapelet_transform_strategy_random():
    X, _ = load_gun_point()
    assert_almost_equal(
        RandomShapeletTransform(random_state=1).fit_transform(X),
        ShapeletTransform(strategy="random", random_state=1).fit_transform(X),
    )


def test_derivative_transform():
    t = DerivativeTransform(method="slope")
    X = np.array([[1, 7, 3, 4], [4, 2, 6, 7]])
    expected_output = np.array([[3.5, 3.5, -2.75, -2.75], [-0.5, -0.5, 3.25, 3.25]])
    assert_almost_equal(t.fit_transform(X), expected_output)
