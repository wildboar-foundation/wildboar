# Authors: Isak Samsten
# License: BSD 3 clause
import pytest
from numpy.testing import assert_almost_equal
from sklearn.utils.estimator_checks import check_estimators_pickle

from wildboar.datasets import load_gun_point
from wildboar.linear_model import (
    CastorClassifier,
    DilatedShapeletClassifier,
    HydraClassifier,
    RandomShapeletClassifier,
    RandomShapeletRegressor,
    RocketClassifier,
    RocketRegressor,
)
from wildboar.utils._testing import (
    assert_exhaustive_parameter_checks,
    assert_parameter_checks,
)
from wildboar.utils.estimator_checks import check_estimator


@pytest.mark.parametrize(
    "estimator, skip",
    [
        (RandomShapeletRegressor(n_shapelets=10), []),
        (RandomShapeletClassifier(n_shapelets=10), []),
        (RandomShapeletClassifier(n_shapelets=10), []),
        (CastorClassifier(n_groups=4, n_shapelets=2), []),
        (DilatedShapeletClassifier(n_shapelets=10), []),
        (RocketClassifier(n_kernels=100), []),
        (RocketRegressor(n_kernels=100), []),
        (HydraClassifier(n_groups=4, n_kernels=2), []),
    ],
)
def test_estimator_checks(estimator, skip):
    check_estimator(
        estimator,
        expected_failed_checks={
            "check_sample_weight_equivalence_on_dense_data": "not working",
            "_check_sample_weights_invariance_samples_order": "not working",
        },
    )
    assert_exhaustive_parameter_checks(estimator)
    assert_parameter_checks(
        estimator,
        skip=skip,
    )
