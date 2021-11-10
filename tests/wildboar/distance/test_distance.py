import pytest
from numpy.testing import assert_almost_equal, assert_equal

from wildboar.datasets import load_dataset
from wildboar.distance import (
    paired_distance,
    paired_subsequence_distance,
    pairwise_distance,
    pairwise_subsequence_distance,
)


@pytest.mark.parametrize(
    "r, expected",
    [
        pytest.param(
            0.0,
            [
                [
                    9.44862115045952,
                    8.233150190764853,
                    9.293339537019579,
                    8.684918504908113,
                    4.110899010402057,
                ],
                [
                    7.0459917692360285,
                    8.089004876024365,
                    9.693203338390742,
                    8.309887075493558,
                    1.4403655009257892,
                ],
                [
                    6.680197873420652,
                    6.658344132911147,
                    8.019689893011222,
                    7.082917686543071,
                    3.620191236815146,
                ],
            ],
        ),
        pytest.param(
            1.0,
            [
                [
                    2.9831583954146033,
                    4.265599023010943,
                    5.219607373220397,
                    4.487744947315483,
                    0.9254090887540904,
                ],
                [
                    3.1507863403171195,
                    4.438504637974342,
                    5.377807094434376,
                    4.66083452998045,
                    0.5730941885156611,
                ],
                [
                    2.1806556575707536,
                    3.410096002460074,
                    4.3134129556068785,
                    3.6251326980747605,
                    1.8504568621069142,
                ],
            ],
        ),
    ],
)
def test_pairwise_dtw_distance(r, expected):
    x, y = load_dataset("GunPoint", repository="wildboar/ucr-tiny")
    assert_almost_equal(
        pairwise_distance(x[0:3], x[10:15], metric="dtw", metric_params={"r": r}),
        expected,
    )


@pytest.mark.parametrize(
    "r, expected",
    [
        pytest.param(
            0.0,
            [
                10.189318553414651,
                12.135216989056529,
                10.517145480739815,
            ],
            id="r=0.0",
        ),
        pytest.param(
            1.0,
            [
                2.8319246476074187,
                7.559313393620837,
                3.9111923927229193,
            ],
            id="r=1.0",
        ),
    ],
)
def test_paired_dtw_distance(r, expected):
    x, y = load_dataset("GunPoint", repository="wildboar/ucr-tiny")
    assert_almost_equal(
        paired_distance(x[0:3], x[30:33], metric="dtw", metric_params={"r": r}),
        expected,
    )


@pytest.mark.parametrize(
    "metric, metric_params, expected_min_dist, expected_min_ind",
    [
        pytest.param(
            "euclidean",
            None,
            [
                [0.23195688803926817, 0.17188321089000758],
                [0.2674267983857602, 0.15871494464186847],
                [0.3937857247136831, 0.4063928467206117],
                [0.13248433440745758, 0.13046911602986688],
                [1.1448052606769021, 1.2488982882334592],
            ],
            [[108, 116], [21, 22], [106, 106], [114, 113], [2, 2]],
            id="euclidean",
        ),
        pytest.param(
            "scaled_euclidean",
            None,
            [
                [2.7386455797485416, 2.038364843603591],
                [4.358640359464751, 2.809961030159136],
                [3.5622988950840084, 2.9527886868779394],
                [3.032223776553107, 2.571925524686407],
                [3.6662031408030042, 1.914621479718833],
            ],
            [[123, 3], [129, 19], [49, 49], [54, 50], [53, 26]],
            id="scaled_euclidean",
        ),
        pytest.param(
            "dtw",
            {"r": 1.0},
            [
                [0.22001506436908683, 0.1693269004631758],
                [0.25999784868100656, 0.1561472638854919],
                [0.3908331894036726, 0.40533280832526025],
                [0.13213970095179928, 0.12977326096534575],
                [1.1251588562664583, 1.2315231778985767],
            ],
            [[107, 116], [21, 22], [106, 106], [114, 113], [2, 2]],
            id="dtw",
        ),
        pytest.param(
            "scaled_dtw",
            {"r": 1.0},
            [
                [1.3321568849328171, 1.7500375448463599],
                [3.0916513463377413, 1.8913769290869902],
                [1.1870303880302593, 1.7097588566478743],
                [1.7188295859581297, 1.904354845981207],
                [2.854933219112498, 1.6711472926757909],
            ],
            [[126, 3], [23, 21], [52, 48], [54, 51], [128, 27]],
            id="scaled_dtw",
        ),
    ],
)
def test_pairwise_subsequence_distance(
    metric, metric_params, expected_min_dist, expected_min_ind
):
    x, y = load_dataset("GunPoint", repository="wildboar/ucr-tiny")
    min_dist, min_ind = pairwise_subsequence_distance(
        x[[2, 3], 0:20],
        x[40:45],
        metric=metric,
        metric_params=metric_params,
        return_index=True,
    )
    assert_almost_equal(min_dist, expected_min_dist)
    assert_equal(min_ind, expected_min_ind)


@pytest.mark.parametrize(
    "metric, metric_params, expected_min_dist, expected_min_ind",
    [
        pytest.param(
            "euclidean",
            None,
            [0.23195688803926817, 0.15871494464186847, 0.22996935251290676],
            [108, 22, 126],
            id="euclidean",
        ),
        pytest.param(
            "scaled_euclidean",
            None,
            [2.7386455797485416, 2.809961030159136, 2.127943475506045],
            [123, 19, 0],
            id="scaled_euclidean",
        ),
        pytest.param(
            "dtw",
            {"r": 1.0},
            [0.22001506436908683, 0.1561472638854919, 0.16706575311794378],
            [107, 22, 130],
            id="dtw",
        ),
        pytest.param(
            "scaled_dtw",
            {"r": 1.0},
            [1.3321568849328171, 1.8913769290869902, 1.506334150321518],
            [126, 21, 0],
            id="scaled_dtw",
        ),
    ],
)
def test_paired_subsequence_distance(
    metric, metric_params, expected_min_dist, expected_min_ind
):
    x, y = load_dataset("GunPoint", repository="wildboar/ucr-tiny")
    min_dist, min_ind = paired_subsequence_distance(
        x[[2, 3, 8], 0:20],
        x[40:43],
        metric=metric,
        metric_params=metric_params,
        return_index=True,
    )
    assert_almost_equal(min_dist, expected_min_dist)
    assert_equal(min_ind, expected_min_ind)
