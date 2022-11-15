from numpy.testing import assert_almost_equal

from wildboar.datasets import load_two_lead_ecg
from wildboar.distance import pairwise_distance


def test_edr_default():
    X, _ = load_two_lead_ecg()
    desired = [
        [0.5975609756097561, 0.47560975609756095, 0.6463414634146342],
        [0.08536585365853659, 0.036585365853658534, 0.13414634146341464],
        [0.0975609756097561, 0.25609756097560976, 0.12195121951219512],
    ]

    assert_almost_equal(
        pairwise_distance(X[:3], X[3:6], metric="edr"),
        desired,
    )
