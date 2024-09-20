# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from sklearn.utils.estimator_checks import check_estimators_pickle

from wildboar.linear_model import CastorClassifier, CastorRegressor


def test_univariate_classifier(gun_point):
    X_train, X_test, y_train, y_test = gun_point
    c = CastorClassifier(n_groups=16, n_shapelets=4, random_state=1)
    c.fit(X_train, y_train)
    assert_almost_equal(c.score(X_test, y_test), 1.0, decimal=3)


def test_multivaraite_classifier(ering):
    X_train, X_test, y_train, y_test = ering
    c = CastorClassifier(n_groups=16, n_shapelets=4, random_state=1)
    c.fit(X_train, y_train)
    assert_almost_equal(c.score(X_test, y_test), 0.78, decimal=3)


def test_univariate_regressor(flood_modeling):
    X_train, X_test, y_train, y_test = flood_modeling
    c = CastorRegressor(n_groups=16, n_shapelets=4, random_state=1)
    c.fit(X_train, y_train)
    assert_almost_equal(c.score(X_test, y_test), 0.75, decimal=2)


def test_multivaraite_regressor(appliances_energy):
    X_train, X_test, y_train, y_test = appliances_energy
    c = CastorRegressor(n_groups=16, n_shapelets=4, random_state=1)
    c.fit(X_train, y_train)
    assert_almost_equal(c.score(X_test, y_test), 0.034, decimal=3)


@pytest.mark.parametrize(
    "estimator",
    [
        CastorClassifier(n_groups=16, n_shapelets=4, random_state=1),
        CastorRegressor(n_groups=16, n_shapelets=4, random_state=1),
    ],
)
def test_attributes(estimator, X, y):
    estimator = CastorClassifier(n_groups=16, n_shapelets=4, random_state=1)
    estimator.fit(X, y)

    # fmt: off
    expected_attribute = np.array([
        1.01380388,  0.9959147 ,  0.8565896 ,  0.88411711,  0.77873522,
        0.82300063, -1.17642124, -0.92288283, -1.3157611 , -0.55738447,
       -1.3797115 , -0.72459105, -0.71749105, -0.66629832, -0.639133  ,
       -0.59301866, -0.57781582, -0.59209143, -0.202485  ,  0.93902103
    ])
    # fmt: on
    assert_almost_equal(
        estimator.pipe_[0]["castortransform"].embedding_.attributes[1][1][2][:20],
        expected_attribute,
    )
