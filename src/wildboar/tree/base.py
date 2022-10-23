# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from ..base import BaseEstimator
from ..utils.validation import _num_timesteps


class BaseTree(BaseEstimator):
    """Base class for tree based estimators."""

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

    def decision_path(self, x, check_input=True):
        check_is_fitted(self, attributes="tree_")
        if check_input:
            x = self._validate_data(x, dtype=float, allow_3d=True, reset=False)
        else:
            x = self._validate_force_n_dims(x)

        return self.tree_.decision_path(x)

    def apply(self, x, check_input=True):
        check_is_fitted(self, attributes="tree_")
        if check_input:
            x = self._validate_data(x, dtype=float, allow_3d=True, reset=False)
        else:
            x = self._validate_force_n_dims(x)

        return self.tree_.apply(x)

    def _more_tags(self):
        return {"X_types": ["2darray", "3darray"]}


class TreeRegressorMixin(RegressorMixin):
    """Mixin for regression trees."""

    def fit(self, x, y, sample_weight=None, check_input=True):
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
            Allow to bypass several input checks

        Returns
        -------

        self: object
        """
        if check_input:
            x, y = self._validate_data(x, y, allow_3d=True, dtype=float, y_numeric=True)
        else:
            x = self._validate_force_n_dims(x)
            self.n_timesteps_in_, self.n_dims_in_ = _num_timesteps(x)
            self.n_features_in_ = self.n_timesteps_in_ * self.n_dims_in_

        random_state = check_random_state(self.random_state)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, x, dtype=float)

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
        check_is_fitted(self, ["tree_"])
        if check_input:
            x = self._validate_data(x, allow_3d=True, dtype=float, reset=False)
        else:
            x = self._validate_force_n_dims(x)

        return self.tree_.predict(x).reshape(-1)


class TreeClassifierMixin(ClassifierMixin):
    """Mixin for classification trees."""

    def fit(self, x, y, sample_weight=None, check_input=True):
        """Fit a classification tree.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps)
            The training time series.

        y : array-like of shape (n_samples,)
            The target values

        sample_weight : array-like of shape (n_samples,)
            If `None`, then samples are equally weighted. Splits that would create child
            nodes with net zero or negative weight are ignored while searching for a
            split in each node. Splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        check_input : bool, optional
            Allow to bypass several input checks.

        Returns
        -------

        self: object
        """
        if check_input:
            x, y = self._validate_data(x, y, allow_3d=True, dtype=float)
            check_classification_targets(y)
        else:
            x = self._validate_force_n_dims(x)
            self.n_timesteps_in_, self.n_dims_in_ = _num_timesteps(x)
            self.n_features_in_ = self.n_timesteps_in_ * self.n_dims_in_

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
        proba = self.predict_proba(x, check_input=check_input)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

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
        if check_input:
            x = self._validate_data(x, allow_3d=True, dtype=float, reset=False)
        else:
            x = self._validate_force_n_dims(x)

        return self.tree_.predict(x)
