# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np
import pytest

from wildboar.ensemble import ExtraShapeletTreesClassifier, ExtraShapeletTreesRegressor
from wildboar.utils.estimator_checks import check_estimator


def test_shap_for():
    clf = ExtraShapeletTreesClassifier(n_estimators=1)
    clf.fit(np.zeros((10, 10)), np.ones(10))


def test_check_estimator():
    check_estimator(
        ExtraShapeletTreesClassifier(),
        expected_failed_checks={
            "check_sample_weight_equivalence_on_dense_data": "not working",
        },
    )
    check_estimator(
        ExtraShapeletTreesRegressor(),
        expected_failed_checks={
            "check_sample_weight_equivalence_on_dense_data": "not working",
        },
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
                "coverage_probability",
                "variability",
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
                "coverage_probability",
                "variability",
                "metric",
                "metric_params",
                "criterion",
            ),
        ),
    ],
)
def test_estimator_params(estimator, expected_estimator_params):
    assert set(estimator.estimator_params) == set(expected_estimator_params)
