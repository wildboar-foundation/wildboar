# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from wildboar.datasets import load_two_lead_ecg
from wildboar.distance.dtw import dtw_alignment, dtw_average, dtw_mapping, jeong_weight


@pytest.mark.parametrize(
    "expected_alignment, weight",
    [
        pytest.param(
            [
                [16.0, 41.0, 77.0, 126.0],
                [25.0, 32.0, 57.0, 93.0],
                [29.0, 34.0, 48.0, 73.0],
                [30.0, 33.0, 42.0, 58.0],
            ],
            None,
        ),
        pytest.param(
            [
                [
                    7.60033300033696,
                    19.7878980882317,
                    37.7878980882317,
                    62.900270515958006,
                ],
                [
                    11.987856431979067,
                    15.20066600067392,
                    27.38823108856866,
                    45.38823108856866,
                ],
                [
                    13.987856431979067,
                    16.375379863621173,
                    22.80099900101088,
                    34.98856408890562,
                ],
                [
                    14.500353828463277,
                    15.987856431979067,
                    20.375379863621173,
                    27.975712863958133,
                ],
            ],
            jeong_weight(4, 0.05),
        ),
    ],
)
def test_dtw_alignment(expected_alignment, weight):
    expected_alignment = np.array(expected_alignment)
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, 7, 8])
    actual_alignment = dtw_alignment(x, y, weight=weight)
    assert_almost_equal(actual_alignment, expected_alignment)


@pytest.mark.parametrize(
    "expected_mapping, weight",
    [
        pytest.param(
            [
                [True, False, False, False],
                [True, False, False, False],
                [True, False, False, False],
                [False, True, True, True],
            ],
            None,
        ),
        pytest.param(
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ],
            jeong_weight(4, 0.5),
        ),
    ],
)
def test_dtw_mapping(expected_mapping, weight):
    expected_mapping = np.array(expected_mapping)
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, 7, 8])
    actual_mapping = dtw_mapping(alignment=dtw_alignment(x, y, weight=weight))
    assert_equal(actual_mapping, expected_mapping)


@pytest.mark.parametrize(
    "method, expected_cost", [("mm", 1.3472766992867564), ("ssg", 1.3477183253349283)]
)
def test_dtw_average(method, expected_cost):
    X, _ = load_two_lead_ecg()
    _, actual_cost = dtw_average(X[:3], method=method, return_cost=True, random_state=0)
    assert_almost_equal(actual_cost, expected_cost)
