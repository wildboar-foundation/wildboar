from numpy.testing import assert_almost_equal, assert_array_equal

from wildboar.explain.counterfactual import NiceCounterfactual
from wildboar.linear_model import RocketClassifier


def test_explanation(gun_point):
    X_train, X_test, y_train, y_test = gun_point

    f = RocketClassifier(n_kernels=100, random_state=1)
    c = NiceCounterfactual(n_neighbors=1)

    f.fit(X_train, y_train)

    X_test = X_test[[1, 2]]
    y_desired = [1, 1]

    c.fit(f, X_train, y_train)
    cf = c.explain(X_test, y_desired)
    assert_array_equal(f.predict(cf), [1, 1])
    assert_array_equal(f.predict(X_test), [2, 2])
