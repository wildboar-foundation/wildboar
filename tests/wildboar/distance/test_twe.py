from numpy.testing import assert_almost_equal

from wildboar.datasets import load_two_lead_ecg
from wildboar.distance import pairwise_distance


def test_twe_default():
    X, _ = load_two_lead_ecg()
    desired = [
        [68.35703983290492, 51.94744984030726, 75.37315617892149],
        [21.540885984838013, 17.268663048364207, 30.4370250165165],
        [19.225670489326124, 38.6009835104645, 22.736767564475553],
    ]

    assert_almost_equal(
        pairwise_distance(X[:3], X[3:6], metric="twe"),
        desired,
    )
