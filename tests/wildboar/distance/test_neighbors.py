import numpy as np
from numpy.testing import assert_equal
from wildboar.datasets import load_two_lead_ecg
from wildboar.distance import KMeans, KMedoids, KNeighborsClassifier
from wildboar.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(KNeighborsClassifier())
    check_estimator(KMeans())
    check_estimator(KMedoids())


def test_kmeans_dtw():
    # fmt: off
    expected = np.array(
        [2, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 2, 0, 2, 0, 2, 2, 2, 0, 0, 2, 1,
         2, 2, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 2, 1, 0, 1,
         0, 0, 2, 0, 0, 0]
    )
    # fmt: on

    X, y = load_two_lead_ecg()
    k = KMeans(n_clusters=3, random_state=1, metric="dtw")
    k.fit(X[:50])
    assert k.n_iter_ == 5
    assert_equal(k.labels_, expected)


def test_kmeans_euclidean():
    # fmt: off
    expected = np.array(
        [2, 2, 2, 2, 0, 0, 1, 2, 1, 0, 2, 1, 0, 0, 1, 1, 2, 2, 1, 0, 1, 0,
         0, 2, 1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 2, 1, 1, 0, 2, 0, 1, 0, 2, 0,
         1, 1, 1, 0, 2, 2]
    )
    # fmt: on

    X, y = load_two_lead_ecg()
    k = KMeans(n_clusters=3, random_state=1, metric="euclidean")
    k.fit(X[:50])
    assert k.n_iter_ == 5
    assert_equal(k.labels_, expected)
