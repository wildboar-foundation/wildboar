# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest

from wildboar.utils.validation import check_array


def test_check_array_multivariate():
    x = np.arange(10 * 3 * 10).reshape(10, 3, 10)
    x_checked = check_array(x, allow_3d=True)
    assert x.dtype == x_checked.dtype

    with pytest.raises(ValueError):
        check_array(x, allow_3d=False)


def test_check_array_ensure_min_dims():
    x = np.arange(10 * 10).reshape(10, 10)
    with pytest.raises(ValueError):
        check_array(x, allow_3d=True, ensure_min_dims=2)

    x = np.arange(3 * 3 * 3).reshape(3, 3, 3)
    check_array(x, allow_3d=True, ensure_min_dims=2)


def test_check_array_allow_3d():
    x = np.arange(10 * 2 * 2 * 10).reshape(10, 2, 2, 10)
    with pytest.raises(ValueError, match="Found array with dim 4. None expected <= 3."):
        check_array(x, allow_3d=True)


def test_check_array_allow_nd():
    x = np.arange(10 * 2 * 2 * 10).reshape(10, 2, 2, 10)
    check_array(x, allow_nd=True)

    x = np.arange(10)
    with pytest.raises(ValueError):
        check_array(x, allow_nd=True)


def test_check_array_order():
    x = np.arange(10 * 3 * 10).reshape(10, 3, 10, order="f")
    x_checked = check_array(x, allow_3d=True)
    assert x_checked.flags.carray

    x_checked = check_array(x, allow_3d=True, order=None)
    assert not x_checked.flags.carray


def test_check_array_ensure2d():
    x = np.arange(10)
    check_array(x, ensure_2d=False)

    x = np.arange(10 * 10).reshape(10, 10)
    check_array(x, ensure_2d=False)


def test_check_array_ravel_1d():
    x = np.arange(10)
    assert check_array(x.reshape(10, 1), ravel_1d=True, ensure_2d=False).ndim == 1
    assert check_array(x.reshape(10, 1), ensure_2d=False).ndim == 2
    assert check_array(x, ensure_2d=False).ndim == 1

    with pytest.raises(ValueError):
        check_array(x.reshape(5, 2), ravel_1d=True)


def test_check_array_force_all_finite():
    with pytest.raises(ValueError, match=".* NaN.*"):
        check_array(
            [1, 2, 3, np.inf, np.nan, -np.inf],
            ensure_2d=False,
            force_all_finite=True,
            allow_eos=True,
        )

    with pytest.raises(ValueError, match=".*infinity.*"):
        check_array(
            [1, 2, 3, np.inf, np.nan, -np.inf],
            ensure_2d=False,
            force_all_finite="allow-nan",
            allow_eos=True,
        )

    check_array(
        [1, 2, 3, np.inf, np.nan, -np.inf], ensure_2d=False, force_all_finite=False
    )

    check_array([1, 2, 3, -np.inf], ensure_2d=False, allow_eos=True)
