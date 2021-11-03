import pytest
import numpy as np
from wildboar.utils import check_dataset


def test_check_1darray():
    x = np.random.random(10)
    x_checked = check_dataset(x, allow_1d=True)
    assert x_checked.shape == (1, 10)
    assert x_checked.dtype == float
