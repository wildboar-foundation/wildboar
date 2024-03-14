# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from sklearn.utils.estimator_checks import check_estimators_pickle
from wildboar.datasets import load_ering
from wildboar.linear_model import RandomShapeletClassifier, RandomShapeletRegressor


def test_univariate(X_train, X_test, y_train, y_test):
    c = RandomShapeletClassifier(n_shapelets=100, random_state=1)
    c.fit(X_train, y_train)
    assert_almost_equal(c.score(X_test, y_test), 0.993, decimal=3)


def test_multivaraite():
    X_train, X_test, y_train, y_test = load_ering(merge_train_test=False)
    c = RandomShapeletClassifier(n_shapelets=100, random_state=4)
    c.fit(X_train, y_train)
    assert_almost_equal(c.score(X_test, y_test), 0.737, decimal=3)


def test_attributes(X, y):
    s = RandomShapeletClassifier(
        n_shapelets=10, random_state=1, max_shapelet_size=0.1
    ).fit(X, y)
    # fmt: off
    expected_attribute = np.array([
        -0.76308721, -0.72875935, -0.67479664, -0.62751573, -0.59122419,
        -0.56505603, -0.56959325, -0.58967131, -0.61996657, -0.64917147,
        -0.65672177, -0.63746166, -0.62916708, -0.60846925, -0.59058785
    ])
    # fmt: on
    assert_almost_equal(
        s.pipe_["transform"].embedding_.attributes[0][1][1], expected_attribute
    )
