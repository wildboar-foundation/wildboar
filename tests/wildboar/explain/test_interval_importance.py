import numpy as np
from numpy.testing import assert_almost_equal

from wildboar.datasets import load_two_lead_ecg
from wildboar.explain import IntervalImportance
from wildboar.tree import ShapeletTreeClassifier


def test_consistent_importance():
    x_train, x_test, y_train, y_test = load_two_lead_ecg(False)
    clf = ShapeletTreeClassifier(n_shapelets=1000, random_state=123)
    clf.fit(x_train, y_train)

    imp = IntervalImportance(scoring="accuracy", random_state=123)
    imp.fit(clf, x_test, y_test)

    actual_scores = imp.importances_.mean
    expected_scores = np.array(
        [
            0.006672519754170314,
            0.016330114135206287,
            0.00913081650570673,
            0.22036874451273042,
            0.23072870939420537,
            0.21580333625987702,
            0.061106233538191355,
            0.027041264266900768,
            0.02546093064091306,
            0.006672519754170292,
        ]
    )

    assert_almost_equal(actual_scores, expected_scores)
