# Authors: Isak Samsten
# License: BSD 3 clause
import pytest
from numpy.testing import assert_almost_equal
from sklearn.utils.estimator_checks import check_estimators_pickle
from wildboar.datasets import load_gun_point
from wildboar.transform import (
    PAA,
    SAX,
    CastorTransform,
    DilatedShapeletTransform,
    HydraTransform,
    IntervalTransform,
    MatrixProfileTransform,
    PivotTransform,
    RandomShapeletTransform,
    RocketTransform,
    ShapeletTransform,
)
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
        (RandomShapeletTransform(), []),
        (ShapeletTransform(), []),
        (RocketTransform(), ["max_size"]),
        (MatrixProfileTransform(), []),
        (PivotTransform(), ["metric_factories"]),
        (SAX(), []),
        (PAA(), []),
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
