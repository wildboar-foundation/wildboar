# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from sklearn.utils.estimator_checks import check_estimators_pickle
from wildboar.datasets import load_ering
from wildboar.linear_model import HydraClassifier


def test_univariate(X_train, X_test, y_train, y_test):
    c = HydraClassifier(n_groups=16, n_kernels=4, random_state=1)
    c.fit(X_train, y_train)
    assert_almost_equal(c.score(X_test, y_test), 1.0, decimal=3)


def test_multivaraite():
    X_train, X_test, y_train, y_test = load_ering(merge_train_test=False)
    c = HydraClassifier(n_groups=16, n_kernels=4, random_state=1)
    c.fit(X_train, y_train)
    assert_almost_equal(c.score(X_test, y_test), 0.8259, decimal=3)


def test_attributes(X, y):
    c = HydraClassifier(n_groups=16, n_kernels=4, random_state=1)
    c.fit(X, y)

    # fmt: off
    expected_attribute = np.array([
        0.04351778,  0.02463252, -0.04244673,  0.42249731,  0.01803031,
       -0.02848694, -0.34004121, -0.07053044, -0.02717261, -0.07860268,
       -0.23437244,  0.12298226, -0.13810706,  0.08698306, -0.00052739,
        0.08321929,  0.12811366,  0.0303113 ,  0.17296394,  0.12579831,
       -0.01444631, -0.0418691 ,  0.12514359, -0.14526094, -0.10594537,
       -0.05486595, -0.06151817, -0.09162781,  0.00766572, -0.02412785,
        0.24755345,  0.13410625,  0.11357846, -0.18445969, -0.11425243,
       -0.08843608
    ])
    # fmt: on
    assert_almost_equal(
        c.pipe_[0]["hydratransform"].embedding_.attributes[0][1], expected_attribute
    )
