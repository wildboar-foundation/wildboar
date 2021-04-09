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

import math

import numpy as np
from numpy.testing import assert_allclose, assert_, assert_equal
from sklearn.utils import check_random_state

from wildboar.datasets.outlier import MinorityLabeler, EmmottLabeler


def _new_labeler_data(n_samples, n_timestep, classes, n_classes):
    random_state = check_random_state(123)
    x = random_state.randn(n_samples, n_timestep)
    y = np.repeat(classes, n_classes)
    return x, y


def _run_labeler(labeler, x, y):
    new_x, new_y = labeler.fit_transform(x, y)
    new_x_1, new_y_1 = labeler.transform(x, y)
    return new_x, new_x_1, new_y, new_y_1


class SetupLabeler:
    def test_binary(self):
        x, y = _new_labeler_data(20, 10, [1, 2], [15, 5])
        self._run_labeler_suite(x, y, *(np.unique(y, return_counts=True)))

    def test_multiclass(self):
        x, y = _new_labeler_data(20, 10, [1, 2, 3], [8, 7, 5])
        self._run_labeler_suite(x, y, *(np.unique(y, return_counts=True)))

    def _run_labeler_suite(self, x, y, classes, label_count):
        self._run_labeler_n_outliers(x, y, classes, label_count, 0.05)
        self._run_labeler_n_outliers(x, y, classes, label_count, 0.1)

    def _run_labeler_n_outliers(self, x, y, classes, label_count, n_outliers):
        new_x, new_x_1, new_y, new_y_1 = _run_labeler(
            self._new_labeler(n_outliers), x, y
        )
        assert_allclose(new_x, new_x_1)
        assert_allclose(new_y, new_y_1)

        self._assert_labeler(new_x, new_y, classes, label_count, n_outliers)

    def _new_labeler(self, n_outliers):
        raise NotImplemented()

    def _assert_labeler(self, new_x, new_y, classes, label_count, n_outliers):
        raise NotImplemented()


class TestMinorityLabeler(SetupLabeler):
    def _new_labeler(self, n_outliers):
        return MinorityLabeler(n_outliers=n_outliers, random_state=1)

    def _assert_labeler(self, x, y, classes, label_count, n_outliers):
        minority_label_index = np.argmin(label_count)
        minority_label = classes[minority_label_index]
        majority_label_count = np.sum(label_count[classes != minority_label])
        actual_n_outliers = math.ceil(majority_label_count * n_outliers)

        n_samples = majority_label_count + actual_n_outliers
        y_outlier = -np.ones(actual_n_outliers)
        y_inlier = np.ones(majority_label_count)

        assert_equal(x.shape, [n_samples, x.shape[1]])
        assert_equal(y.shape, [n_samples])
        assert_allclose(y, np.concatenate([y_outlier, y_inlier]))


class TestEmmottLabeler(SetupLabeler):
    def _new_labeler(self, n_outliers):
        return EmmottLabeler(n_outliers=n_outliers, difficulty="any", random_state=1)

    def _assert_labeler(self, x, y, classes, label_count, n_outliers):
        pass
