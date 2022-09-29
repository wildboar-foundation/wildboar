# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from wildboar.datasets import load_dataset
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.explain.counterfactual import PrototypeCounterfactual, counterfactuals


@pytest.mark.parametrize(
    "clf, expected_score",
    [
        pytest.param(
            ShapeletForestClassifier(n_estimators=10, random_state=123),
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
            np.array([3.062681783939084, 8.38541221454514]),
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
        scoring="euclidean",
        random_state=123,
    )
    assert_almost_equal(actual_score, expected_score)


def test_counterfactuals_prototype():
    x_train, x_test, y_train, y_test = load_dataset(
        "GunPoint", repository="wildboar/ucr-tiny", merge_train_test=False
    )
    clf = ShapeletForestClassifier(n_estimators=10, random_state=123)
    clf.fit(x_train, y_train)

    method = PrototypeCounterfactual(method="shapelet", metric="euclidean")
    _, _, actual_score = counterfactuals(
        clf,
        x_test[1:3],
        1,
        method=method,
        train_x=x_train,
        train_y=y_train,
        scoring="euclidean",
        random_state=123,
    )
    expected_score = np.array([0.8731226473373813, 6.270890036587185])
    assert_almost_equal(actual_score, expected_score)
