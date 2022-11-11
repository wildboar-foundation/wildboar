from numpy.testing import assert_almost_equal

from wildboar.datasets import load_two_lead_ecg
from wildboar.distance import pairwise_distance


def test_lcss_default():
    X, _ = load_two_lead_ecg()
    desired = [
        [0.012195121951219523, 0.03658536585365857, 0.04878048780487809],
        [0.0, 0.024390243902439046, 0.024390243902439046],
        [0.0, 0.03658536585365857, 0.024390243902439046],
    ]

    assert_almost_equal(
        pairwise_distance(X[:3], X[3:6], metric="lcss"),
        desired,
    )
