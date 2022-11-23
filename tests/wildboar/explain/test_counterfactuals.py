# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier

from wildboar.datasets import load_dataset
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.explain.counterfactual import (
    KNeighborsCounterfactual,
    PrototypeCounterfactual,
    ShapeletForestCounterfactual,
    counterfactuals,
)
from wildboar.utils.estimator_checks import check_estimator


def test_estimator_check_shapelet_forest_counterfactual():
    check_estimator(ShapeletForestCounterfactual(), skip_scikit=True)


@pytest.mark.parametrize(
    "clf, expected_score",
    [
        pytest.param(
            ShapeletForestClassifier(n_shapelets=10, n_estimators=10, random_state=123),
            np.array([1.0216381495890032, 5.925046545971925]),
        ),
        pytest.param(
            KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
            np.array([2.307078907707276, 9.355327387515722]),
        ),
        pytest.param(
            KNeighborsClassifier(n_neighbors=1, metric="euclidean"),
            np.array([2.6514561178753313, 8.378653689114698]),
        ),
        pytest.param(
            RandomForestClassifier(random_state=123),
            np.array([3.401708, 7.8155463]),
        ),
    ],
)
def test_counterfactuals_best(clf, expected_score):
    x_train, x_test, y_train, y_test = load_dataset(
        "GunPoint", repository="wildboar/ucr-tiny", merge_train_test=False
    )
    clf.fit(x_train, y_train)
    _, _, actual_score = counterfactuals(
        clf,
        x_test[1:3],
        1,
        method="best",
        train_x=x_train,
        train_y=y_train,
        proximity="euclidean",
        random_state=123,
    )
    assert_almost_equal(actual_score, expected_score)


def test_counterfactuals_prototype():
    x_train, x_test, y_train, y_test = load_dataset(
        "GunPoint", repository="wildboar/ucr-tiny", merge_train_test=False
    )
    clf = ShapeletForestClassifier(n_shapelets=10, n_estimators=10, random_state=123)
    clf.fit(x_train, y_train)

    method = PrototypeCounterfactual(method="shapelet", metric="euclidean")
    _, _, actual_score = counterfactuals(
        clf,
        x_test[1:3],
        1,
        method=method,
        train_x=x_train,
        train_y=y_train,
        proximity="euclidean",
        random_state=123,
    )
    expected_score = np.array([2.3592611, 6.2593162])
    assert_almost_equal(actual_score, expected_score)


@pytest.mark.parametrize(
    "counterfactual, estimator",
    [
        (ShapeletForestCounterfactual(), ShapeletForestClassifier(n_shapelets=10)),
        (KNeighborsCounterfactual(), KNeighborsClassifier()),
        (PrototypeCounterfactual(), ShapeletForestClassifier()),
    ],
)
def test_counterfactual_not_fitted_estimator(counterfactual, estimator):
    with pytest.raises(NotFittedError):
        counterfactual.fit(estimator, np.zeros((3, 3)), np.ones(3))
