# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from wildboar.datasets import load_dataset
from wildboar.datasets.outlier import minority_outliers


@pytest.mark.parametrize(
    "n_outliers, expected_n_outliers, "
    "expected_first_outlier_sample, expected_outlier_norm",
    [
        (0.05, 5, 18, 27.294688068996287),
        (0.1, 10, 18, 38.60051804670847),
        (0.2, 20, 18, 54.58937618813437),
    ],
)
def test_minority_outliers(
    n_outliers,
    expected_n_outliers,
    expected_first_outlier_sample,
    expected_outlier_norm,
):
    x, y = load_dataset("GunPoint", repository="wildboar/ucr-tiny")
    x_outlier, y_outlier = minority_outliers(
        x, y, n_outliers=n_outliers, random_state=123
    )
    assert_outliers(
        x,
        x_outlier,
        y_outlier,
        expected_n_outliers=expected_n_outliers,
        expected_first_outlier_sample=expected_first_outlier_sample,
        expected_outlier_norm=expected_outlier_norm,
    )


def assert_outliers(
    x,
    x_outlier,
    y_outlier,
    *,
    expected_n_outliers,
    expected_first_outlier_sample,
    expected_outlier_norm
):
    actual_n_outliers = np.where(y_outlier == -1)[0].shape[0]
    actual_fist_outlier_sample = (np.sum(x_outlier[0] - x, axis=1) == 0).argmax()
    actual_outlier_norm = np.linalg.norm(x_outlier[np.where(y_outlier == -1)])
    assert_almost_equal(actual_outlier_norm, expected_outlier_norm)
    assert_equal(actual_fist_outlier_sample, expected_first_outlier_sample)
    assert_equal(actual_n_outliers, expected_n_outliers)
