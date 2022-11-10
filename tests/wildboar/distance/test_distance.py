# Authors: Isak Samsten
# License: BSD 3 clause

import pytest
from numpy.testing import assert_almost_equal, assert_equal

from wildboar.datasets import load_gun_point
from wildboar.distance import (
    paired_distance,
    paired_subsequence_distance,
    pairwise_distance,
    pairwise_subsequence_distance,
    subsequence_match,
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
    x, y = load_gun_point()
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
    x, y = load_gun_point()
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
    x, y = load_gun_point()
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
    x, y = load_gun_point()
    min_dist, min_ind = paired_subsequence_distance(
        x[[2, 3, 8], 0:20],
        x[40:43],
        metric=metric,
        metric_params=metric_params,
        return_index=True,
    )
    assert_almost_equal(min_dist, expected_min_dist)
    assert_equal(min_ind, expected_min_ind)


@pytest.mark.parametrize(
    "metric, metric_params, expected_indicies, expected_dists",
    [
        [
            "euclidean",
            {},
            [
                [126, 127, 125, 128, 129, 130, 124, 131, 132, 123],
                [109, 110, 108, 111, 138, 118, 117, 124, 119, 112],
            ],
            [
                [
                    0.03397965606582041,
                    0.0343842104203779,
                    0.0351323095527136,
                    0.03539658207798578,
                    0.03645583018956124,
                    0.036979988389006124,
                    0.03703322511321164,
                    0.03831106381122556,
                    0.04012265023031157,
                    0.04104258653686285,
                ],
                [
                    0.18783105110608192,
                    0.1962122726590744,
                    0.19969665170517434,
                    0.21358058104153474,
                    0.2286672665170618,
                    0.23024263523833352,
                    0.2310810578541956,
                    0.2323035020498981,
                    0.23286074160686906,
                    0.23352672398649205,
                ],
            ],
        ],
        [
            "scaled_euclidean",
            {},
            [
                [90, 89, 83, 91, 84, 82, 85, 88, 81, 92],
                [75, 89, 88, 87, 71, 90, 86, 76, 70, 72],
            ],
            [
                [
                    0.5171341149025254,
                    0.6106690561241724,
                    0.6199414822706423,
                    0.6389336609057809,
                    0.6861095369454515,
                    0.6919642938064638,
                    0.7518358747794223,
                    0.7551918338623284,
                    0.8089276966112849,
                    0.8305581115853162,
                ],
                [
                    0.5395380605368274,
                    0.5994517516083463,
                    0.6060069282120507,
                    0.6572386395006412,
                    0.6865909258011326,
                    0.7388046565921842,
                    0.7530885210085916,
                    0.810795863777238,
                    0.8376980330763003,
                    0.8501658606924453,
                ],
            ],
        ],
        [
            "dtw",
            {"r": 0.5},
            [
                [126, 127, 125, 128, 129, 130, 124, 131, 132, 123],
                [109, 110, 108, 125, 124, 116, 111, 117, 126, 118],
            ],
            [
                [
                    0.03377513034186882,
                    0.03417138314023305,
                    0.0351323095527136,
                    0.035180236615724184,
                    0.03620168568738906,
                    0.03692724340632057,
                    0.037000014085456216,
                    0.03829773823788508,
                    0.04012265023031157,
                    0.04092858833956632,
                ],
                [
                    0.1859017766089558,
                    0.19554091347043767,
                    0.19601154714547592,
                    0.2090317042574604,
                    0.21199763600964866,
                    0.2124628268335682,
                    0.21347216622866322,
                    0.2137223700636814,
                    0.2152617975710072,
                    0.21833681247621853,
                ],
            ],
        ],
        [
            "scaled_dtw",
            {"r": 0.5},
            [
                [90, 84, 86, 92, 83, 87, 89, 85, 88, 91],
                [75, 72, 71, 89, 90, 88, 74, 87, 91, 77],
            ],
            [
                [
                    0.5171341149025254,
                    0.53438801033894,
                    0.5449097439782348,
                    0.5815724724804437,
                    0.5868331003736715,
                    0.5874821636080592,
                    0.5972173148623663,
                    0.5978139524706985,
                    0.613990263275831,
                    0.6158445792495065,
                ],
                [
                    0.40792177885373476,
                    0.49151144102128486,
                    0.5507076093071009,
                    0.5574072024166825,
                    0.5643688082122642,
                    0.5774972865470762,
                    0.5904850517319787,
                    0.6204986495558397,
                    0.6239597262924556,
                    0.6674760578321446,
                ],
            ],
        ],
    ],
)
def test_subseqence_match_default(
    metric, metric_params, expected_indicies, expected_dists
):
    x, _ = load_gun_point()
    subsequence = x[0, 3:15]
    actual_indices, actual_dists = subsequence_match(
        subsequence,
        x[1:3],
        return_distance=True,
        metric=metric,
        metric_params=metric_params,
    )

    for actual_dist, expected_dist in zip(actual_dists, expected_dists):
        assert_almost_equal(actual_dist, expected_dist)

    for actual_index, expected_index in zip(actual_indices, expected_indicies):
        assert_equal(actual_index, expected_index)


@pytest.mark.parametrize(
    "left, right",
    [
        ["euclidean", "dtw"],
        ["scaled_euclidean", "scaled_dtw"],
        ["scaled_euclidean", "mass"],
        ["mass", "scaled_dtw"],
    ],
)
def test_subsequence_match_equivalent(left, right):
    x, _ = load_gun_point()
    subsequence = x[10, 40:60]

    left_indicies, left_distances = subsequence_match(
        subsequence,
        x[0:9],
        return_distance=True,
        max_matches=20,
        exclude=4,
        metric=left,
        metric_params=dict(r=0.0) if "dtw" in left else {},
    )

    right_indicies, right_distances = subsequence_match(
        subsequence,
        x[0:9],
        return_distance=True,
        max_matches=20,
        exclude=4,
        metric=right,
        metric_params=dict(r=0.0) if "dtw" in right else {},
    )

    for left_distance, right_distance in zip(left_distances, right_distances):
        assert_almost_equal(left_distance, right_distance)

    for left_index, right_index in zip(left_indicies, right_indicies):
        assert_equal(left_index, right_index)
