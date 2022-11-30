import numpy as np
import pytest

from wildboar.utils._misc import _test_ts_array
from wildboar.utils.validation import _check_ts_array


def test_check_tsarray_skip_2D():
    X = np.arange(10 * 10).reshape(10, 10)
    X = X[::2, :]
    X_checked = _check_ts_array(X)
    assert X.base is not X_checked.base


def test_check_tsarray_skip_3D():
    X = np.arange(10 * 3 * 10).reshape(10, 3, 10)
    X = X[::2, :, :]
    X_checked = _check_ts_array(X)
    assert X.base is not X_checked.base
    assert X_checked.dtype == float


def test_check_tsarray_slice_3D():
    X = np.zeros((10, 4, 10))
    X_sliced = X[::2, ::2, :4:2]
    assert 0 == _test_ts_array(_check_ts_array(X_sliced))

    with pytest.raises(
        ValueError,
        match="Buffer and memoryview are not contiguous in the same dimension",
    ):
        _test_ts_array(X_sliced)


def test_check_tsarray_slice_same_base_3D():
    X = np.zeros((10, 3, 10))
    X_sliced = X[::2, :, :3]
    assert _check_ts_array(X_sliced).base is X

    X_sliced = X[::2, ::2, ::2]
    assert _check_ts_array(X_sliced).base is not X
