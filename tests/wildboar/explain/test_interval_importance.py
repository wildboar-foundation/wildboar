import numpy as np
from numpy.testing import assert_almost_equal

from wildboar.datasets import load_two_lead_ecg
from wildboar.explain import AmplitudeImportance, IntervalImportance, ShapeletImportance
from wildboar.tree import ShapeletTreeClassifier


def test_interval_importance():
    x_train, x_test, y_train, y_test = load_two_lead_ecg(False)
    clf = ShapeletTreeClassifier(n_shapelets=1000, random_state=123)
    clf.fit(x_train, y_train)

    imp = IntervalImportance(scoring="accuracy", random_state=123)
    imp.fit(clf, x_test, y_test)

    actual_scores = imp.importances_.mean
    # fmt: off
    expected_scores = np.array([0.00438982, 0.01299385, 0.02809482, 0.2500439 , 0.21738367,
                                0.21071115, 0.00298507, 0.        , 0.        , 0.        ])
    # fmt: on

    assert_almost_equal(actual_scores, expected_scores)


def test_amplitude_importance():
    x_train, x_test, y_train, y_test = load_two_lead_ecg(False)
    clf = ShapeletTreeClassifier(n_shapelets=1000, random_state=123)
    clf.fit(x_train, y_train)

    imp = AmplitudeImportance(scoring="accuracy", random_state=123)
    imp.fit(clf, x_test, y_test)

    actual_scores = imp.importances_.mean
    expected_scores = np.array([0.30640913, 0.20895522, 0.08165057, 0.0])

    assert_almost_equal(actual_scores, expected_scores)


def test_shapelet_importance():
    x_train, x_test, y_train, y_test = load_two_lead_ecg(False)
    clf = ShapeletTreeClassifier(n_shapelets=1000, random_state=123)
    clf.fit(x_train, y_train)

    imp = ShapeletImportance(scoring="accuracy", random_state=123)
    imp.fit(clf, x_test, y_test)

    actual_scores = imp.importances_.mean
    # fmt: off
    expected_scores = np.array(
        [0.38366989, 0.3046532 , 0.17559263, 0.37928007, 0.26777875,
         0.34855136, 0.24231782, 0.35381914, 0.13081651, 0.0403863 ]
    )
    # fmt: on

    assert_almost_equal(actual_scores, expected_scores)
