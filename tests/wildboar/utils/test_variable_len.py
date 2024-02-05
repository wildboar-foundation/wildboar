import numpy as np
import pytest
import wildboar
from wildboar.utils.variable_len import (
    eos,
    get_variable_length,
    is_end_of_series,
    is_variable_length,
)


def test_is_variable_len():
    x1 = np.array([1, 2, 3, eos])
    assert is_variable_length(x1)
    assert is_variable_length(x1.astype(np.float32))

    x2 = np.array([1, 2, 3, eos], dtype=np.float32)
    assert is_variable_length(x2)
    assert is_variable_length(x2.astype(np.float64))


@pytest.mark.parametrize(
    "x",
    [
        np.array(
            [
                [eos, eos, np.nan, np.inf],
                [1, eos, -np.inf, eos],
            ],
            dtype=np.float64,
        ),
    ],
)
def test_is_end_of_series(x):
    expected = np.array(
        [
            [True, True, False, False],
            [False, True, False, True],
        ]
    )
    np.testing.assert_equal(is_end_of_series(x), expected)
    np.testing.assert_equal(is_end_of_series(x.astype(np.float32)), expected)
    # TODO(1.3): Remove
    expected[1, 2] = True
    np.testing.assert_equal(wildboar.iseos(x), expected)


def test_is_end_of_series_2d():
    x = np.array([[1, 2, 3, eos], [1, 2, eos, eos], [1, 2, 3, 4]])
    lengths = get_variable_length(x)
    np.testing.assert_equal(lengths, np.array([3, 2, 4]))


def test_is_end_of_series_1d():
    assert get_variable_length(np.array([1, 2, 3, eos])) == 3
    assert get_variable_length([eos, 0, 0, 0]) == 0


def test_is_end_of_series_3d():
    x = np.array(
        [
            [[1, np.nan, 3, eos], [1, 2, eos, eos], [1, 2, 3, 4]],
            [[1, 2, 3, eos], [1, 2, eos, eos], [eos, eos, 3, 4]],
        ]
    )
    lengths = get_variable_length(x)
    np.testing.assert_equal(lengths, np.array([[3, 2, 4], [3, 2, 0]]))
