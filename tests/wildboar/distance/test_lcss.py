import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from wildboar.datasets import load_two_lead_ecg
from wildboar.distance import (
    pairwise_distance,
    pairwise_subsequence_distance,
    subsequence_match,
)


@pytest.fixture
def X():
    X, _ = load_two_lead_ecg()
    return X


def test_lcss_default(X):
    desired = [
        [0.012195121951219523, 0.03658536585365857, 0.04878048780487809],
        [0.0, 0.024390243902439046, 0.024390243902439046],
        [0.0, 0.03658536585365857, 0.024390243902439046],
    ]

    assert_almost_equal(
        pairwise_distance(X[:3], X[3:6], metric="lcss"),
        desired,
    )


def test_lcss_subsequence_distance(X):
    desired = [
        [0.0, 0.0, 0.0],
        [0.06000000000000005, 0.06000000000000005, 0.06000000000000005],
        [0.040000000000000036, 0.040000000000000036, 0.040000000000000036],
        [0.040000000000000036, 0.040000000000000036, 0.09999999999999998],
        [0.0, 0.0, 0.0],
    ]

    actual = pairwise_subsequence_distance(
        X[:3, :50], X[50:55], metric="lcss", metric_params={"r": 1.0}
    )
    assert_almost_equal(actual, desired)


def test_lcss_subsequence_match(X):
    desired_inds = [
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ]

    desired_dists = [
        np.array([0.02, 0.04, 0.06, 0.08, 0.1]),
        np.array([0.04, 0.06, 0.08, 0.1]),
        np.array([0.08, 0.06, 0.04, 0.02, 0.0, 0.0, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1]),
    ]

    actual_inds, actual_dists = subsequence_match(
        X[3, :50], X[52:55], metric="lcss", threshold=0.1, return_distance=True
    )

    for actual_ind, desired_ind in zip(actual_inds, desired_inds):
        assert_almost_equal(actual_ind, desired_ind)

    for actual_dist, desired_dist in zip(actual_dists, desired_dists):
        assert_almost_equal(actual_dist, desired_dist)
