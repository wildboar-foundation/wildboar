# Authors: Isak Samsten
# License: BSD 3 clause

import sys
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from ..base import BaseEstimator
from ..utils.validation import _check_ts_array, _num_timesteps


# noqa: PR01
class BaseTree(BaseEstimator, metaclass=ABCMeta):
    """Base class for tree based estimators."""

    _parameter_constraints: dict = {
        "max_depth": [
            Interval(Integral, 1, sys.getrecursionlimit(), closed="left"),
            None,
        ],
        "min_samples_split": [Interval(Integral, 2, None, closed="left")],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left")],
        "min_impurity_decrease": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _validate_not_check_input(self, x, reset=True):
        x = self._validate_force_n_dims(x)
        if reset:
            self.n_timesteps_in_, self.n_dims_in_ = _num_timesteps(x)
            self.n_features_in_ = self.n_timesteps_in_ * self.n_dims_in_

        return _check_ts_array(x)

    @abstractmethod
    def _fit(self, X, y, sample_weight, max_depth, random_state):
        pass

    def decision_path(self, x, check_input=True):
        """
        Compute the decision path of the tree.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep) or\
 (n_samples, n_dims, n_timestep)
            The input samples.
        check_input : bool, optional
            Bypass array validation. Only set to True if you are sure your data
            is valid.

        Returns
        -------
        sparse matrix of shape (n_samples, n_nodes)
            An indicator array where each nonzero values indicate that the sample
            traverses a node.
        """
        check_is_fitted(self, attributes="tree_")
        if check_input:
            x = self._validate_data(
                x, dtype=float, ensure_ts_array=True, allow_3d=True, reset=False
            )
        else:
            x = self._validate_not_check_input(x, reset=False)

        return self.tree_.decision_path(x)

    def apply(self, x, check_input=True):
        """
        Return the index of the leaf that each sample is predicted by.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep) or\
 (n_samples, n_dims, n_timestep)
            The input samples.
        check_input : bool, optional
            Bypass array validation. Only set to True if you are sure your data
            is valid.

        Returns
        -------
        ndarray of shape (n_samples, )
            For every sample, return the index of the leaf that the sample
            ends up in. The index is in the range [0; node_count].

        Examples
        --------
        Get the leaf probability distribution of a prediction:

        >>> from wildboar.datasets import load_gun_point
        >>> from wildboar.tree import ShapeletTreeClassifier
        >>> X, y = load_gun_point()
        >>> tree = ShapeletTreeClassifier()
        >>> tree.fit(X, y)
        >>> leaves = tree.apply(X)
        >>> tree.tree_.value.take(leaves, axis=0)
        array([[0., 1.],
               [0., 1.],
               [1., 0.]])

        This is equvivalent to using `tree.predict_proba`.
        """
        check_is_fitted(self, attributes="tree_")
        if check_input:
            x = self._validate_data(
                x, dtype=float, ensure_ts_array=True, allow_3d=True, reset=False
            )
        else:
            x = self._validate_not_check_input(x, reset=False)

        return self.tree_.apply(x)

    def _more_tags(self):
        return {"X_types": ["2darray", "3darray"]}


class BaseTreeRegressor(RegressorMixin, BaseTree, metaclass=ABCMeta):
    def fit(self, x, y, sample_weight=None, check_input=True):
        """
        Fit the estimator.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps)
            The training time series.
        y : array-like of shape (n_samples,)
            Target values as floating point values.
        sample_weight : array-like of shape (n_samples,), optional
            If `None`, then samples are equally weighted. Splits that would create child
            nodes with net zero or negative weight are ignored while searching for a
            split in each node. Splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        check_input : bool, optional
            Allow to bypass several input checks.

        Returns
        -------
        self
            This object.

        """
        self._validate_params()
        if check_input:
            x, y = self._validate_data(
                x, y, allow_3d=True, ensure_ts_array=True, dtype=float, y_numeric=True
            )
        else:
            x = self._validate_not_check_input(x)

        random_state = check_random_state(self.random_state)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, x, dtype=float)

        self._fit(
            x,
            y,
            sample_weight,
            min(self.max_depth or sys.getrecursionlimit(), sys.getrecursionlimit()),
            random_state,
        )
        return self

    def predict(self, x, check_input=True):
        """
        Predict the value of x.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps)
            The input time series.
        check_input : bool, optional
            Allow to bypass several input checking. Don't use this parameter unless you
            know what you do.

        Returns
        -------
        ndarray of shape (n_samples,)
            The predicted classes.

        """
        check_is_fitted(self, ["tree_"])
        if check_input:
            x = self._validate_data(
                x, allow_3d=True, ensure_ts_array=True, dtype=float, reset=False
            )
        else:
            x = self._validate_not_check_input(x, reset=False)

        return self.tree_.predict(x).reshape(-1)


class BaseTreeClassifier(ClassifierMixin, BaseTree, metaclass=ABCMeta):
    """Mixin for classification trees."""

    def fit(self, x, y, sample_weight=None, check_input=True):
        """
        Fit a classification tree.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps)
            The training time series.
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,), optional
            If `None`, then samples are equally weighted. Splits that would create child
            nodes with net zero or negative weight are ignored while searching for a
            split in each node. Splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        check_input : bool, optional
            Allow to bypass several input checks.

        Returns
        -------
        self
            This instance.

        """
        self._validate_params()
        if check_input:
            x, y = self._validate_data(
                x, y, allow_3d=True, ensure_ts_array=True, dtype=float
            )
            check_classification_targets(y)
        else:
            x = self._validate_not_check_input(x)

        if hasattr(self, "class_weight") and self.class_weight is not None:
            class_weight = compute_sample_weight(self.class_weight, y)
        else:
            class_weight = None

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
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

        self._fit(
            x,
            y,
            sample_weight,
            min(self.max_depth or sys.getrecursionlimit(), sys.getrecursionlimit()),
            random_state,
        )
        return self

    def predict(self, x, check_input=True):
        """
        Predict the regression of the input samples x.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps)
            The input time series.
        check_input : bool, optional
            Allow to bypass several input checking. Don't use this parameter unless you
            know what you do.

        Returns
        -------
        ndarray of shape (n_samples,)
            The predicted classes.

        """
        proba = self.predict_proba(x, check_input=check_input)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, x, check_input=True):
        """
        Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        x :  array-like of shape (n_samples, n_timesteps)
            The input time series.
        check_input : bool, optional
            Allow to bypass several input checking. Don't use this parameter unless you
            know what you do.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the classes
            corresponds to that in the attribute `classes_`.

        """
        check_is_fitted(self, ["tree_"])
        if check_input:
            x = self._validate_data(
                x, ensure_ts_array=True, allow_3d=True, dtype=float, reset=False
            )
        else:
            x = self._validate_not_check_input(x, reset=False)

        return self.tree_.predict(x)
