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
