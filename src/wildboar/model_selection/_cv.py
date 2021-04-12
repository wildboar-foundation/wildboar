# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten

import math
from abc import ABCMeta
import warnings

import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection._split import _build_repr
from sklearn.utils.validation import check_array


class RepeatedOutlierSplit(metaclass=ABCMeta):
    """Repeated random outlier cross-validator

    Yields indicies that split the dataset into training and test sets.

    Note
    ----
    Contrary to other cross-validation strategies, the random outlier
    cross-validator does not ensure that all folds will be different.
    Instead, the inlier samples are shuffled and new outlier samples
    are inserted in the training and test sets repeatedly.
    """

    def __init__(
        self,
        n_splits=None,
        *,
        test_size=0.2,
        n_outlier=0.05,
        shuffle=True,
        random_state=None,
    ):
        """Construct a new cross-validator

        Parameters
        ----------
        n_splits : int, optional
            The maximum number of splits.

            - if None, the number of splits is determined by the number of
              outliers as, `total_n_outliers/(n_inliers * n_outliers)`

            - if int, the number of splits is an upper-bound

        test_size : float, optional
            The size of the test set.

        n_outlier : float, optional
            The fraction of outliers in the training and test sets.

        shuffle : bool, optional
            Shuffle the training indicies in each iteration.

        random_state : int or RandomState, optional
            The psudo-random number generator
        """
        self.n_outliers = n_outlier
        self.test_size = test_size
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X, y, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            The samples
        y : object
            The labels
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        outlier_index = (y == -1).nonzero()[0]
        inlier_index = (y == 1).nonzero()[0]
        total_n_outliers = len(outlier_index)
        n_outliers = min(
            total_n_outliers, math.ceil(inlier_index.shape[0] * self.n_outliers)
        )
        if self.n_splits is None:
            n_splits = total_n_outliers // n_outliers
        else:
            n_splits = min(self.n_splits, total_n_outliers // n_outliers)
        return n_splits

    def split(self, x, y, groups=None):
        """Return training and test indicies

        Parameters
        ----------
        x : object
            Always ignored, exists for compatibility.
        y : object
            The labels
        groups : object, optional
            Always ignored, exists for compatibility.

        Yields
        -------
        train_idx, test_idx : ndarray
            The training and test indicies
        """
        y = check_array(y, ensure_2d=False)
        random_state = check_random_state(self.random_state)
        outlier_index = (y == -1).nonzero()[0]
        inlier_index = (y == 1).nonzero()[0]
        total_n_outliers = len(outlier_index)
        n_inliers = inlier_index.shape[0]
        n_outliers = min(
            total_n_outliers, math.ceil(inlier_index.shape[0] * self.n_outliers)
        )
        if n_outliers < 2:
            warnings.warn("to few outliers", UserWarning)

        if self.n_splits is None:
            n_splits = total_n_outliers // n_outliers
        else:
            n_splits = min(self.n_splits, total_n_outliers // n_outliers)
        random_state.shuffle(outlier_index)

        for i in range(n_splits):
            if self.shuffle:
                random_state.shuffle(inlier_index)
            outlier_index_sample = outlier_index[
                (i * n_outliers) : (i * n_outliers + n_outliers)
            ]
            outlier_test_index = math.ceil(n_outliers * self.test_size)
            inlier_test_index = math.ceil(n_inliers * self.test_size)

            yield (
                np.concatenate(
                    [
                        inlier_index[inlier_test_index:],  # last elements
                        outlier_index_sample[outlier_test_index:],
                    ]
                ),
                np.concatenate(
                    [
                        inlier_index[:inlier_test_index],  # first elements
                        outlier_index_sample[:outlier_test_index],
                    ]
                ),
            )

    def __repr__(self):
        return _build_repr(self)
