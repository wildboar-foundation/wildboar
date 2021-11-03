import numpy as np
import pytest
from wildboar.utils import check_array


def test_check_array_multivariate():
    x = np.arange(10 * 3 * 10).reshape(10, 3, 10)
    x_checked = check_array(x, allow_multivariate=True)
    assert x.dtype == x_checked.dtype

    with pytest.raises(ValueError):
        check_array(x, allow_multivariate=False)


def test_check_array_contiguous():
    x = np.arange(10 * 3 * 10).reshape(10, 3, 10, order="f")
    x_checked = check_array(x, allow_multivariate=True)
    assert x_checked.flags.carray

    x_checked = check_array(x, allow_multivariate=True, contiguous=False)
    assert not x_checked.flags.carray
