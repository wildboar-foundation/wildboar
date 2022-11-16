from numpy.testing import assert_almost_equal

from wildboar.datasets import load_two_lead_ecg
from wildboar.distance import pairwise_distance


def test_msm_default():
    X, _ = load_two_lead_ecg()
    desired = [
        [35.57120902929455, 28.885248728096485, 40.931326208636165],
        [11.86974082980305, 12.198534050490707, 17.675189964473248],
        [10.293032762594521, 23.823819940909743, 14.209565471857786],
    ]

    assert_almost_equal(
        pairwise_distance(X[:3], X[3:6], metric="msm"),
        desired,
    )
