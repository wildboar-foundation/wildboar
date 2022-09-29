# Authors: Isak Samsten
# License: BSD 3 clause

import pytest

from wildboar.ensemble import ExtraShapeletTreesClassifier, ExtraShapeletTreesRegressor
from wildboar.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(
        ExtraShapeletTreesClassifier(), ignore=["check_sample_weights_invariance"]
    )
    check_estimator(
        ExtraShapeletTreesRegressor(), ignore=["check_sample_weights_invariance"]
    )


@pytest.mark.parametrize(
    "estimator, expected_estimator_params",
    [
        (
            ExtraShapeletTreesClassifier(),
            (
                "max_depth",
                "min_samples_leaf",
                "min_impurity_decrease",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "metric",
                "metric_params",
                "criterion",
            ),
        ),
        (
            ExtraShapeletTreesRegressor(),
            (
                "max_depth",
                "min_impurity_decrease",
                "min_samples_leaf",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "metric",
                "metric_params",
                "criterion",
            ),
        ),
    ],
)
def test_estimator_params(estimator, expected_estimator_params):
    assert set(estimator.estimator_params) == set(expected_estimator_params)
