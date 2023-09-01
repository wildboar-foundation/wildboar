# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_equal
from wildboar.datasets import load_gun_point
from wildboar.ensemble import ProximityForestClassifier
from wildboar.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(
        ProximityForestClassifier(n_estimators=10),
        ignore=["check_sample_weights_invariance"],
    )
    check_estimator(
        ProximityForestClassifier(n_estimators=10),
        ignore=["check_sample_weights_invariance"],
    )


@pytest.fixture
def clf():
    X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)
    clf = ProximityForestClassifier(n_estimators=3, random_state=2)
    clf.fit(X_train, y_train)
    return clf


@pytest.fixture
def dataset():
    return load_gun_point(merge_train_test=False)


def test_fit_predict(clf, dataset):
    _, X_test, _, _ = dataset
    y_actual = clf.predict(X_test)
    y_desired = np.array(
        [
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            1.0,
            2.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            1.0,
            2.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            1.0,
            2.0,
            1.0,
            1.0,
            1.0,
            2.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
            1.0,
            2.0,
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            2.0,
            2.0,
            1.0,
        ],
        dtype=float,
    )

    assert_equal(y_actual, y_desired)


def test_branch(clf):
    desired = [
        np.array(
            [
                [1, 2, 3, -1, 5, -1, 7, -1, 9, -1, -1, 12, -1, -1, 15, -1, 17, -1, -1],
                [
                    14,
                    11,
                    4,
                    -1,
                    6,
                    -1,
                    8,
                    -1,
                    10,
                    -1,
                    -1,
                    13,
                    -1,
                    -1,
                    16,
                    -1,
                    18,
                    -1,
                    -1,
                ],
            ]
        ),
        np.array(
            [
                [
                    1,
                    2,
                    3,
                    4,
                    -1,
                    6,
                    -1,
                    8,
                    -1,
                    -1,
                    -1,
                    12,
                    -1,
                    14,
                    -1,
                    -1,
                    17,
                    -1,
                    19,
                    -1,
                    21,
                    -1,
                    -1,
                ],
                [
                    16,
                    11,
                    10,
                    5,
                    -1,
                    7,
                    -1,
                    9,
                    -1,
                    -1,
                    -1,
                    13,
                    -1,
                    15,
                    -1,
                    -1,
                    18,
                    -1,
                    20,
                    -1,
                    22,
                    -1,
                    -1,
                ],
            ]
        ),
        np.array(
            [
                [
                    1,
                    2,
                    3,
                    4,
                    -1,
                    6,
                    -1,
                    8,
                    -1,
                    -1,
                    -1,
                    12,
                    -1,
                    -1,
                    15,
                    16,
                    -1,
                    -1,
                    19,
                    20,
                    21,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    14,
                    11,
                    10,
                    5,
                    -1,
                    7,
                    -1,
                    9,
                    -1,
                    -1,
                    -1,
                    13,
                    -1,
                    -1,
                    18,
                    17,
                    -1,
                    -1,
                    24,
                    23,
                    22,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
            ]
        ),
    ]

    for estimator, desired_branch in zip(clf.estimators_, desired):
        assert_equal(estimator.tree_.branch, desired_branch)