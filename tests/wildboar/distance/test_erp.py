from numpy.testing import assert_almost_equal

from wildboar.datasets import load_two_lead_ecg
from wildboar.distance import pairwise_distance


def test_erp_default():
    X, _ = load_two_lead_ecg()
    desired = [
        [35.701564208604395, 23.691653557121754, 38.59651095978916],
        [10.988020450808108, 4.921718027442694, 13.570069573819637],
        [10.255016681738198, 19.7473459597677, 9.590815722942352],
    ]

    assert_almost_equal(
        pairwise_distance(X[:3], X[3:6], metric="erp"),
        desired,
    )
