import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from wildboar.base import is_explainer
from wildboar.datasets import load_two_lead_ecg
from wildboar.explain import AmplitudeImportance, IntervalImportance, ShapeletImportance
from wildboar.tree import ShapeletTreeClassifier
from wildboar.utils._testing import (
    assert_exhaustive_parameter_checks,
    assert_parameter_checks,
)


@pytest.mark.parametrize(
    "explainer",
    [
        IntervalImportance(),
        AmplitudeImportance(),
        ShapeletImportance(),
    ],
)
def test_parameter_constrains(explainer):
    assert_exhaustive_parameter_checks(explainer)
    assert_parameter_checks(explainer)


@pytest.mark.parametrize(
    "explainer",
    [
        IntervalImportance(),
        AmplitudeImportance(),
        ShapeletImportance(),
    ],
)
def test_is_explainer(explainer):
    assert is_explainer(explainer)


def test_interval_importance():
    x_train, x_test, y_train, y_test = load_two_lead_ecg(False)
    clf = ShapeletTreeClassifier(n_shapelets=1000, random_state=123)
    clf.fit(x_train, y_train)

    imp = IntervalImportance(scoring="accuracy", random_state=123)
    imp.fit(clf, x_test, y_test)

    actual_scores = imp.importances_.mean
    expected_scores = np.array(
        [
            0.0064969,
            0.0149254,
            0.01036,
            0.2363477,
            0.2361721,
            0.2186128,
            0.0612818,
            0.0273924,
            0.0231782,
            0.0061457,
        ]
    )

    assert_almost_equal(actual_scores, expected_scores)


def test_amplitude_importance():
    x_train, x_test, y_train, y_test = load_two_lead_ecg(False)
    clf = ShapeletTreeClassifier(n_shapelets=1000, random_state=123)
    clf.fit(x_train, y_train)

    imp = AmplitudeImportance(scoring="accuracy", random_state=123)
    imp.fit(clf, x_test, y_test)

    actual_scores = imp.importances_.mean
    expected_scores = np.array([0.2712906, 0.191396, 0.1018437, 0.0570676])

    assert_almost_equal(actual_scores, expected_scores)


def test_shapelet_importance():
    x_train, x_test, y_train, y_test = load_two_lead_ecg(False)
    clf = ShapeletTreeClassifier(n_shapelets=1000, random_state=123)
    clf.fit(x_train, y_train)

    imp = ShapeletImportance(scoring="accuracy", random_state=123)
    imp.fit(clf, x_test, y_test)

    actual_scores = imp.importances_.mean
    expected_scores = np.array(
        [
            0.3028973,
            0.2431958,
            0.1931519,
            0.2835821,
            0.2756804,
            0.260755,
            0.2186128,
            0.2941176,
            0.1729587,
            -0.0087796,
        ]
    )

    assert_almost_equal(actual_scores, expected_scores)
