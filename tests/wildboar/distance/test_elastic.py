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


@pytest.mark.parametrize(
    "metric", ["dtw", "erp", "lcss", "msm", "twe", "ddtw", "wdtw", "wddtw"]
)
def test_benchmark(benchmark, metric, X):
    X, y = load_two_lead_ecg()
    x = X[:100].reshape(-1).copy()
    y = X[100:200].reshape(-1).copy()

    benchmark(pairwise_distance, x, y, metric=metric, metric_params={"r": 1.0})


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


def test_lcss_elastic(X):
    desired = [
        [0.05555555555555558, 0.02777777777777779, 0.0],
        [0.04166666666666663, 0.01388888888888884, 0.0],
        [0.02777777777777779, 0.01388888888888884, 0.0],
    ]

    actual = pairwise_distance(X[15:18, 10:], X[22:25], metric="lcss")
    assert_almost_equal(actual, desired)


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


def test_erp_elastic(X):
    desired = [
        [11.895995151251554, 17.091679483652115, 7.37759099714458],
        [12.14670612406917, 17.1011259064544, 12.251492985291407],
        [10.75756766833365, 11.364691082388163, 17.637203577905893],
    ]

    actual = pairwise_distance(X[15:18, 10:-10], X[22:25], metric="erp")
    assert_almost_equal(actual, desired)


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


def test_msm_elastic(X):
    desired = [
        [27.870645819231868, 31.660631390288472, 25.051930798217654],
        [28.350629466818646, 32.191368132131174, 29.716226652963087],
        [26.691123254597187, 26.367045141756535, 32.39279904589057],
    ]

    actual = pairwise_distance(X[15:18, 10:-10], X[22:25], metric="msm")
    assert_almost_equal(actual, desired)


def test_twe_default(X):
    desired = [
        [68.35703983290492, 51.94744984030726, 75.37315617892149],
        [21.540885984838013, 17.268663048364207, 30.4370250165165],
        [19.225670489326124, 38.6009835104645, 22.736767564475553],
    ]

    assert_almost_equal(
        pairwise_distance(X[:3], X[3:6], metric="twe"),
        desired,
    )


def test_twe_elastic(X):
    desired = [
        [36.95036049121615, 44.13520717373491, 30.747075087815496],
        [36.83389494967832, 43.321251590881424, 38.2953548782132],
        [33.929289467781814, 30.94000878250598, 42.822785688996255],
    ]

    actual = pairwise_distance(X[15:18, 10:-10], X[22:25], metric="twe")
    assert_almost_equal(actual, desired)


def test_dtw_elastic(X):
    desired = [
        [1.3184709191755355, 2.2190483689290956, 1.321736156689961],
        [1.792996058064275, 2.6451060425383344, 2.1773549928211353],
        [1.712739739550159, 2.329732794256892, 2.209988047866857],
    ]

    actual = pairwise_distance(X[15:18, 10:-10], X[22:25], metric="dtw")
    assert_almost_equal(actual, desired)


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


def test_edr_elastic(X):
    desired = [
        [0.07317073170731707, 0.13414634146341464, 0.13414634146341464],
        [0.14634146341463414, 0.17073170731707318, 0.21951219512195122],
        [0.12195121951219512, 0.15853658536585366, 0.3048780487804878],
    ]

    actual = pairwise_distance(X[15:18, 10:-10], X[22:25], metric="edr")
    assert_almost_equal(actual, desired)


def test_edr_subsequence_distance(X):
    desired = [
        [0.16666666666666666, 0.2, 0.23333333333333334],
        [0.3333333333333333, 0.2, 0.06666666666666667],
        [0.5, 0.16666666666666666, 0.16666666666666666],
        [0.06666666666666667, 0.36666666666666664, 0.43333333333333335],
        [0.5, 0.16666666666666666, 0.26666666666666666],
    ]

    actual = pairwise_subsequence_distance(
        X[:3, 20:50], X[50:55], metric="edr", metric_params={"r": 1.0}
    )
    assert_almost_equal(actual, desired)


def test_edr_subsequence_match(X):
    desired_inds = [
        np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
        None,
        np.array([18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]),
    ]

    desired_dists = [
        np.array(
            [
                0.23333333,
                0.2,
                0.16666667,
                0.13333333,
                0.1,
                0.06666667,
                0.1,
                0.13333333,
                0.16666667,
                0.2,
                0.23333333,
            ]
        ),
        None,
        np.array(
            [
                0.23333333,
                0.2,
                0.16666667,
                0.13333333,
                0.1,
                0.06666667,
                0.03333333,
                0.0,
                0.03333333,
                0.06666667,
                0.1,
                0.13333333,
                0.16666667,
                0.2,
                0.23333333,
            ]
        ),
    ]
    actual_inds, actual_dists = subsequence_match(
        X[3, 20:50], X[52:55], metric="edr", threshold=0.25, return_distance=True
    )

    for actual_ind, desired_ind in zip(actual_inds, desired_inds):
        if desired_ind is None:
            assert actual_ind is None
        else:
            assert_almost_equal(actual_ind, desired_ind)

    for actual_dist, desired_dist in zip(actual_dists, desired_dists):
        if desired_dist is None:
            assert actual_dist is None
        else:
            assert_almost_equal(actual_dist, desired_dist)


def test_twe_subsequence_distance(X):
    desired = [
        [11.649268791079521, 12.180933441966772, 10.074451576930286],
        [13.404799362820386, 9.72582231733799, 7.091040290904044],
        [19.939912437385317, 9.372541546821594, 8.709062487578393],
        [7.400111306458712, 13.362499579787254, 16.208067022275923],
        [19.014237593103942, 7.727498351947966, 12.209753487698737],
    ]

    actual = pairwise_subsequence_distance(
        X[:3, 20:50], X[50:55], metric="twe", metric_params={"r": 1.0}
    )
    assert_almost_equal(actual, desired)


def test_twe_subsequence_match(X):
    desired_inds = [np.array([17, 18, 19, 20]), None, np.array([24, 25, 26, 27])]

    desired_dists = [
        np.array([8.13667472, 6.23308036, 7.37104807, 9.2250953]),
        None,
        np.array([7.83366694, 5.89159112, 7.10912424, 8.91684203]),
    ]
    actual_inds, actual_dists = subsequence_match(
        X[3, 20:50], X[52:55], metric="twe", threshold=10, return_distance=True
    )

    for actual_ind, desired_ind in zip(actual_inds, desired_inds):
        if desired_ind is None:
            assert actual_ind is None
        else:
            assert_almost_equal(actual_ind, desired_ind)

    for actual_dist, desired_dist in zip(actual_dists, desired_dists):
        if desired_dist is None:
            assert actual_dist is None
        else:
            assert_almost_equal(actual_dist, desired_dist)
