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


def test_dtw_default(X):
    desired = [
        [3.5537758584340273, 2.4239916042161553, 4.014102144666234],
        [1.1052711252130065, 0.6155502223457359, 1.3603006745123651],
        [0.8904511837689495, 1.7621925843838169, 0.9011076709882901],
    ]

    actual = pairwise_distance(X[:3], X[3:6], metric="dtw")
    assert_almost_equal(actual, desired)


def test_dtw_elastic(X):
    desired = [
        [1.3184709191755355, 2.2190483689290956, 1.321736156689961],
        [1.792996058064275, 2.6451060425383344, 2.1773549928211353],
        [1.712739739550159, 2.329732794256892, 2.209988047866857],
    ]

    actual = pairwise_distance(X[15:18, 10:-10], X[22:25], metric="dtw")
    assert_almost_equal(actual, desired)


def test_wdtw_default(X):
    desired = [
        [1.2552015453334, 0.840657845575087, 1.397156676128907],
        [0.38125602031080913, 0.21652193787948482, 0.4816084512219569],
        [0.3074476369386968, 0.6205562983854465, 0.32328277683306095],
    ]

    actual = pairwise_distance(X[:3], X[3:6], metric="wdtw")
    assert_almost_equal(actual, desired)


def test_ddtw_default(X):
    desired = [
        [0.8813726884722624, 0.845112685537995, 0.900431284098585],
        [0.5410051293098898, 0.2817966461521348, 0.5468055721838778],
        [0.48507795268129195, 0.5978382928900522, 0.6375355302587586],
    ]

    actual = pairwise_distance(X[:3], X[3:6], metric="ddtw")
    assert_almost_equal(actual, desired)


def test_wddtw_default(X):
    desired = [
        [0.31308613461101015, 0.31035907083813735, 0.32896305445930935],
        [0.1899586497011622, 0.10291886213457994, 0.20003786125651232],
        [0.17004678165807582, 0.22040991627513995, 0.2315062579605518],
    ]

    actual = pairwise_distance(X[:3], X[3:6], metric="wddtw")
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


def test_msm_subsequence_distance(X):
    desired = [
        [5.884132720530033, 6.212927086278796, 6.066322663798928],
        [7.452256789430976, 5.099573530256748, 4.565707825124264],
        [10.069798408076167, 4.704355984926224, 5.372886322438717],
        [3.7369808349758387, 6.707286648452282, 8.716198019683361],
        [9.660764277447015, 4.539345602039248, 7.1159256673417985],
    ]

    actual = pairwise_subsequence_distance(
        X[:3, 20:50], X[50:55], metric="msm", metric_params={"r": 1.0}
    )
    assert_almost_equal(actual, desired)


def test_msm_subsequence_match(X):
    desired_inds = [
        np.array([15, 16, 17, 18, 19, 20, 21]),
        np.array([18]),
        np.array([22, 23, 24, 25, 26, 27, 28]),
    ]

    desired_dists = [
        np.array(
            [
                9.51183589,
                7.39649074,
                5.25149395,
                3.18999668,
                4.36803799,
                6.29684289,
                8.28040772,
            ]
        ),
        np.array([9.40306032]),
        np.array(
            [
                9.56636097,
                7.28114184,
                5.03819322,
                3.00207221,
                4.56476857,
                6.47269578,
                8.46184267,
            ]
        ),
    ]
    actual_inds, actual_dists = subsequence_match(
        X[3, 20:50], X[52:55], metric="msm", threshold=10, return_distance=True
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


def test_erp_subsequence_distance(X):
    desired = [
        [5.860492743551731, 5.51637732796371, 6.681755607947707],
        [7.235501637682319, 4.641821198165417, 4.494721375405788],
        [9.578443309292197, 4.325426779687405, 5.447692297399044],
        [3.5930565763264894, 6.707286648452282, 8.571289233863354],
        [9.660764277447015, 4.137808752711862, 7.071880591567606],
    ]

    actual = pairwise_subsequence_distance(
        X[:3, 20:50], X[50:55], metric="erp", metric_params={"r": 1.0}
    )
    assert_almost_equal(actual, desired)


def test_erp_subsequence_match(X):
    desired_inds = [
        np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),
        np.array([18, 19]),
        np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
    ]

    desired_dists = [
        np.array(
            [
                9.18203199,
                7.7950432,
                6.75960507,
                5.86078154,
                5.3475072,
                5.04154608,
                4.83975397,
                4.55013021,
                3.87872966,
                3.14008564,
                2.99880227,
                3.94245976,
                5.0746254,
                6.12780273,
                7.33895668,
                8.69492247,
                9.91924116,
            ]
        ),
        np.array([9.40306032, 9.81500911]),
        np.array(
            [
                8.6444893,
                7.40741205,
                6.27264785,
                5.55448708,
                5.16029088,
                4.8895849,
                4.73896403,
                4.25164569,
                3.56806188,
                3.00207221,
                3.44339784,
                4.47218099,
                5.59572043,
                6.69218668,
                7.81572618,
                9.00694879,
            ]
        ),
    ]

    actual_inds, actual_dists = subsequence_match(
        X[3, 20:50], X[52:55], metric="erp", threshold=10, return_distance=True
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


def test_dtw_subsequence_distance(X):
    desired = [
        [0.9892175168657907, 1.0063214093961415, 0.728809497908996],
        [1.1535877966795345, 0.8751305858765588, 0.5918762820384382],
        [1.6463501354826122, 0.8111886944554642, 0.7541254649350714],
        [0.9060083944619378, 1.2025019188465507, 1.0572433054156687],
        [1.6611303630194554, 0.7583611765873354, 0.8452636056135603],
    ]

    actual = pairwise_subsequence_distance(
        X[:3, 20:50], X[50:55], metric="dtw", metric_params={"r": 1.0}
    )

    assert_almost_equal(actual, desired)


def test_dtw_subsequence_match(X):
    desired_inds = [
        np.array([17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
        None,
        np.array([23, 24, 25, 26, 27, 28, 29, 30, 31, 32]),
    ]

    desired_dists = [
        np.array(
            [
                0.69910004,
                0.55788568,
                0.55933766,
                0.54274111,
                0.55801145,
                0.54481298,
                0.56354989,
                0.64684609,
                0.67375639,
                0.96500658,
            ]
        ),
        None,
        np.array(
            [
                0.817363,
                0.56456229,
                0.51808363,
                0.49816442,
                0.49318993,
                0.50482461,
                0.50767,
                0.51700958,
                0.5434501,
                0.66714102,
            ]
        ),
    ]

    actual_inds, actual_dists = subsequence_match(
        X[3, 20:50], X[52:55], metric="dtw", threshold=1.0, return_distance=True
    )

    print(actual_inds)
    print(actual_dists)

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


def test_wdtw_subsequence_distance(X):
    desired = [
        [0.33976588576901373, 0.35696941021281137, 0.29529477646405033],
        [0.40743928135924934, 0.2968451811061202, 0.20069638850403906],
        [0.5718888312347068, 0.27778330485873565, 0.2602774731189233],
        [0.3073448804124781, 0.4393089961212446, 0.43653337575076473],
        [0.5747838846608478, 0.25904902500981475, 0.2936395717131312],
    ]

    actual = pairwise_subsequence_distance(
        X[:3, 20:50], X[50:55], metric="wdtw", metric_params={"r": 1.0}
    )
    assert_almost_equal(actual, desired)


def test_wdtw_subsequence_match(X):
    desired_inds = [
        np.array([17, 18, 19, 20, 21, 22, 23, 24, 25]),
        None,
        np.array([23, 24, 25, 26, 27, 28, 29, 30, 31, 32]),
    ]

    desired_dists = [
        np.array(
            [
                0.2407963,
                0.19020777,
                0.19224316,
                0.1907945,
                0.19902954,
                0.1989307,
                0.20893769,
                0.23934628,
                0.25257523,
            ]
        ),
        None,
        np.array(
            [
                0.28205538,
                0.19499937,
                0.17651201,
                0.16970836,
                0.17094885,
                0.17784312,
                0.18213986,
                0.18870032,
                0.20072309,
                0.24603989,
            ]
        ),
    ]

    actual_inds, actual_dists = subsequence_match(
        X[3, 20:50], X[52:55], metric="wdtw", threshold=0.3, return_distance=True
    )

    print(actual_inds)
    print(actual_dists)

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


def test_ddtw_subsequence_distance(X):
    desired = [
        [0.601288821783055, 0.5221424348860837, 0.3430566541234427],
        [0.5805189571399302, 0.4800321547177683, 0.24825602966276156],
        [0.721921564327584, 0.37348317637184164, 0.5336120107986725],
        [0.7454798658395534, 0.6940890375635499, 0.4371010970146026],
        [0.7298777256758102, 0.4691353842406821, 0.40021046946845795],
    ]

    actual = pairwise_subsequence_distance(
        X[:3, 20:50], X[50:55], metric="ddtw", metric_params={"r": 1.0}
    )
    print(actual.tolist())
    assert_almost_equal(actual, desired)


def test_wddtw_subsequence_distance(X):
    desired = [
        [0.2101076600612728, 0.18757147836726834, 0.11904901540274042],
        [0.2019113352931962, 0.1691148360424211, 0.08709842622519594],
        [0.25230204368944487, 0.12932437487520157, 0.18590896498560244],
        [0.25993385996374907, 0.24188499844845562, 0.15344920657661343],
        [0.2533634606527797, 0.16450048045815122, 0.1455145521526342],
    ]

    actual = pairwise_subsequence_distance(
        X[:3, 20:50], X[50:55], metric="wddtw", metric_params={"r": 1.0}
    )
    print(actual.tolist())
    assert_almost_equal(actual, desired)
