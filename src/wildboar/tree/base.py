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
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from wildboar.utils import check_array


class BaseTree(BaseEstimator):
    """Base class for tree based estimators."""

    def __init__(
        self,
        *,
        force_dim=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
    ):
        self.force_dim = force_dim
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _validate_x_predict(self, x, check_input):
        if check_input:
            x = check_array(x, allow_multivariate=True)

        if isinstance(self.force_dim, int):
            x = np.reshape(x, [x.shape[0], self.force_dim, -1])

        if x.shape[-1] != self.n_timestep_:
            raise ValueError(
                "illegal input shape ({} != {})".format(x.shape[-1], self.n_timestep_)
            )
        if x.ndim > 2 and x.shape[1] != self.n_dims_:
            raise ValueError(
                "illegal input shape ({} != {}".format(x.shape[1], self.n_dims_)
            )

        return x

    def decision_path(self, x, check_input=True):
        check_is_fitted(self, attributes="tree_")
        x = self._validate_x_predict(x, check_input)
        return self.tree_.decision_path(x)

    def apply(self, x, check_input=True):
        check_is_fitted(self, attributes="tree_")
        x = self._validate_x_predict(x, check_input)
        return self.tree_.apply(x)


class TreeRegressorMixin(RegressorMixin):
    """Mixin for regression trees."""

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a shapelet tree regressor from the training set

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            The training time series.

        y : array-like of shape (n_samples,)
            Target values as floating point values

        sample_weight : array-like of shape (n_samples,)
            If `None`, then samples are equally weighted. Splits that would create child
            nodes with net zero or negative weight are ignored while searching for a
            split in each node. Splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        check_input : bool, optional
            Allow to bypass several input checking. Don't use this parameter unless you
            know what you do.

        Returns
        -------

        self: object
        """
        if check_input:
            X = check_array(X, allow_multivariate=True, dtype=float)
            y = check_array(y, ensure_2d=False, dtype=float)

        n_samples = X.shape[0]
        if isinstance(self.force_dim, int):
            X = np.reshape(X, [n_samples, self.force_dim, -1])

        n_timesteps = X.shape[-1]

        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels={} does not match "
                "number of samples={}".format(len(y), n_samples)
            )

        self.n_timestep_ = n_timesteps
        self.n_dims_ = n_dims
        random_state = check_random_state(self.random_state)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=float)

        self._fit(X, y, sample_weight, random_state)
        return self

    def predict(self, x, check_input=True):
        """Predict the regression of the input samples x.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps)
            The input time series

        check_input : bool, optional
            Allow to bypass several input checking. Don't use this parameter unless you
            know what you do.

        Returns
        -------

        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self, ["tree_"])
        x = self._validate_x_predict(x, check_input)
        return self.tree_.predict(x)


class TreeClassifierMixin(ClassifierMixin):
    """Mixin for classifation trees."""

    def fit(self, x, y, sample_weight=None, check_input=True):
        """Fit a shapelet tree regressor from the training set

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps)
            The training time series.

        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            The target values (class labels) as integers

        sample_weight : array-like of shape (n_samples,)
            If `None`, then samples are equally weighted. Splits that would create child
            nodes with net zero or negative weight are ignored while searching for a
            split in each node. Splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        check_input : bool, optional
            Allow to bypass several input checking. Don't use this parameter unless you
            know what you do.

        Returns
        -------

        self: object
        """
        if check_input:
            x = check_array(x, allow_multivariate=True, dtype=float)
            y = check_array(y, ensure_2d=False)

        n_samples = x.shape[0]
        if isinstance(self.force_dim, int):
            x = np.reshape(x, [n_samples, self.force_dim, -1])

        n_timesteps = x.shape[-1]

        if x.ndim > 2:
            n_dims = x.shape[1]
        else:
            n_dims = 1

        if hasattr(self, "class_weight") and self.class_weight is not None:
            class_weight = compute_sample_weight(self.class_weight, y)
        else:
            class_weight = None

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels={} does not match "
                "number of samples={}".format(len(y), n_samples)
            )

        self.n_classes_ = len(self.classes_)
        self.n_timestep_ = n_timesteps
        self.n_dims_ = n_dims
        random_state = check_random_state(
            self.random_state if hasattr(self, "random_state") else None
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, x, dtype=float)

        if class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * class_weight
            else:
                sample_weight = class_weight

        self._fit(x, y, sample_weight, random_state)
        return self

    def predict(self, x, check_input=True):
        """Predict the regression of the input samples x.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps)
            The input time series

        check_input : bool, optional
            Allow to bypass several input checking. Don't use this parameter unless you
            know what you do.

        Returns
        -------

        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        return self.classes_[
            np.argmax(self.predict_proba(x, check_input=check_input), axis=1)
        ]

    def predict_proba(self, x, check_input=True):
        """Predict class probabilities of the input samples X.  The predicted
        class probability is the fraction of samples of the same class
        in a leaf.

        Parameters
        ----------
        x :  array-like of shape (n_samples, n_timesteps)
            The input time series

        check_input : bool, optional
            Allow to bypass several input checking. Don't use this parameter unless you
            know what you do.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the classes
            corresponds to that in the attribute `classes_`
        """
        check_is_fitted(self, ["tree_"])
        x = self._validate_x_predict(x, check_input)
        return self.tree_.predict(x)
