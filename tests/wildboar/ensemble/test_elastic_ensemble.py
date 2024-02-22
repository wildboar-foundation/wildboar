import numpy as np
import pytest
from wildboar.datasets import load_gun_point
from wildboar.ensemble import ElasticEnsembleClassifier


def test_elastic_ensemble_classifier_fit():
    X, y = load_gun_point()
    X_train = X[:30]
    y_train = y[:30]
    clf = ElasticEnsembleClassifier(
        n_neighbors=1,
        metric={
            "dtw": {"min_r": 0.1, "max_r": 0.3, "num_r": 3},
            "ddtw": {"min_r": 0.1, "max_r": 0.3, "num_r": 3},
        },
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X[30:40])
    np.testing.assert_almost_equal([score[1] for score in clf.scores_], [0.9, 1.0])
    np.testing.assert_equal(
        proba.argmax(axis=1),
        [0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    )
