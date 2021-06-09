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

import sys
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.validation import _check_sample_weight, check_array, check_is_fitted

from ..embed._rocket import _SAMPLING_METHOD, RocketFeatureEngineer
from ..embed._shapelet import RandomShapeletFeatureEngineer
from ._tree_builder import (
    EntropyCriterion,
    ExtraTreeBuilder,
    GiniCriterion,
    MSECriterion,
    Tree,
    TreeBuilder,
)

CLF_CRITERION = {"gini": GiniCriterion, "entropy": EntropyCriterion}
REG_CRITERION = {"mse": MSECriterion}


class BaseTree(BaseEstimator, metaclass=ABCMeta):
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
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(x.ndim))
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
        if check_input:
            x = check_array(x, dtype=np.float64, allow_nd=True, order="C")

        if x.dtype != np.float64 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)
        return x

    def decision_path(self, x, check_input=True):
        check_is_fitted(self, attributes="tree_")
        x = self._validate_x_predict(x, check_input)
        return self.tree_.decision_path(x)

    def apply(self, x, check_input=True):
        check_is_fitted(self, attributes="tree_")
        x = self._validate_x_predict(x, check_input)
        return self.tree_.apply(x)

    @abstractmethod
    def _get_feature_engineer(self):
        pass

    @abstractmethod
    def _get_tree_builder(
        self, x, y, sample_weights, feature_engineer, random_state, max_depth
    ):
        pass

    def _fit(self, x, y, sample_weights, random_state):
        max_depth = (
            sys.getrecursionlimit() if self.max_depth is None else self.max_depth
        )
        if self.min_impurity_decrease < 0.0:
            raise ValueError(
                "min_impurity_decrease must be larger than or equal to 0.0"
            )

        if max_depth <= 0:
            raise ValueError("max_depth must be larger than 0")
        elif max_depth > sys.getrecursionlimit():
            warnings.warn("max_depth exceeds the maximum recursion limit.")

        feature_engineer = self._get_feature_engineer()
        tree_builder = self._get_tree_builder(
            x,
            y,
            sample_weights,
            feature_engineer,
            random_state,
            max_depth,
        )
        tree_builder.build_tree()
        self.tree_ = tree_builder.tree_


class RegressorTreeMixin:
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
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, dtype=np.float64, ensure_2d=False)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions")

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

        if X.dtype != np.float64 or not X.flags.c_contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if y.dtype != np.float64 or not y.flags.c_contiguous:
            y = np.ascontiguousarray(y, dtype=np.float64)

        self.n_timestep_ = n_timesteps
        self.n_dims_ = n_dims
        random_state = check_random_state(self.random_state)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, np.float64)

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

    def _get_tree_builder(
        self, x, y, sample_weights, feature_engineer, random_state, max_depth
    ):
        if self.criterion not in REG_CRITERION:
            raise ValueError("criterion (%s) is not supported" % self.criterion)

        criterion = REG_CRITERION[self.criterion](y)
        tree = Tree(feature_engineer, 1)
        return TreeBuilder(
            x,
            sample_weights,
            feature_engineer,
            criterion,
            tree,
            random_state,
            max_depth=max_depth,
            min_sample_split=self.min_samples_split,
            min_sample_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )


class ClassifierTreeMixin:
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
            x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, ensure_2d=False)

        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions")

        n_samples = x.shape[0]
        if isinstance(self.force_dim, int):
            x = np.reshape(x, [n_samples, self.force_dim, -1])

        n_timesteps = x.shape[-1]

        if x.ndim > 2:
            n_dims = x.shape[1]
        else:
            n_dims = 1

        if self.class_weight is not None:
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

        if x.dtype != np.float64 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)

        if not y.flags.c_contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        self.n_classes_ = len(self.classes_)
        self.n_timestep_ = n_timesteps
        self.n_dims_ = n_dims
        random_state = check_random_state(self.random_state)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, x, np.float64)

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

    def _get_tree_builder(
        self, x, y, sample_weights, feature_engineer, random_state, max_depth
    ):
        if self.criterion not in CLF_CRITERION:
            raise ValueError("criterion (%s) is not supported" % self.criterion)

        criterion = CLF_CRITERION[self.criterion](y, self.n_classes_)
        tree = Tree(feature_engineer, self.n_classes_)
        return TreeBuilder(
            x,
            sample_weights,
            feature_engineer,
            criterion,
            tree,
            random_state,
            max_depth=max_depth,
            min_sample_split=self.min_samples_split,
            min_sample_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )


class BaseShapeletTree(BaseTree):
    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        n_shapelets=10,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        force_dim=None,
        random_state=None,
    ):
        super().__init__(
            force_dim=force_dim,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.random_state = check_random_state(random_state)
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params
        self.n_timestep_ = None
        self.n_dims_ = None

    def _get_feature_engineer(self):
        if (
            self.min_shapelet_size < 0
            or self.min_shapelet_size > self.max_shapelet_size
        ):
            raise ValueError(
                "min_shapelet_size {0} <= 0 or {0} > {1}".format(
                    self.min_shapelet_size, self.max_shapelet_size
                )
            )
        if self.max_shapelet_size > 1:
            raise ValueError("max_shapelet_size %d > 1" % self.max_shapelet_size)

        max_shapelet_size = int(self.n_timestep_ * self.max_shapelet_size)
        min_shapelet_size = int(self.n_timestep_ * self.min_shapelet_size)
        if min_shapelet_size < 2:
            min_shapelet_size = 2

        return RandomShapeletFeatureEngineer(
            self.n_timestep_,
            self.metric,
            self.metric_params,
            min_shapelet_size,
            max_shapelet_size,
            self.n_shapelets,
        )


class ShapeletTreeRegressor(RegressorMixin, RegressorTreeMixin, BaseShapeletTree):
    """A shapelet tree regressor.

    Attributes
    ----------

    tree_ : Tree
    """

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        n_shapelets=10,
        min_shapelet_size=0,
        max_shapelet_size=1,
        metric="euclidean",
        metric_params=None,
        force_dim=None,
        criterion="mse",
        random_state=None,
    ):
        """Construct a shapelet tree regressor

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of the tree. If `None` the tree is expanded until all
            leaves are pure or until all leaves contain less than `min_samples_split`
            samples

        min_samples_split : int, optional
            The minimum number of samples to split an internal node

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf

        criterion : {"mse"}, optional
            The criterion used to evaluate the utility of a split

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value

        n_shapelets : int, optional
            The number of shapelets to sample at each node.

        min_shapelet_size : float, optional
            The minimum length of a sampled shapelet expressed as a fraction, computed
            as `min(ceil(X.shape[-1] * min_shapelet_size), 2)`.

        max_shapelet_size : float, optional
            The maximum length of a sampled shapelet, expressed as a fraction, computed
            as `ceil(X.shape[-1] * max_shapelet_size)`.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Distance metric used to identify the best shapelet.

        metric_params : dict, optional
            Parameters for the distance measure

        force_dim : int, optional
            Force the number of dimensions.

            - If int, force_dim reshapes the input to shape (n_samples, force_dim, -1)
              for interoperability with `sklearn`.

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator;
            - If `RandomState` instance, `random_state` is the random number generator;
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super(ShapeletTreeRegressor, self).__init__(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            n_shapelets=n_shapelets,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            force_dim=force_dim,
            random_state=random_state,
        )
        self.criterion = criterion


class ExtraShapeletTreeRegressor(ShapeletTreeRegressor):
    """An extra shapelet tree regressor.

    Extra shapelet trees are constructed by sampling a distance threshold
    uniformly in the range [min(dist), max(dist)].

    Attributes
    ----------

    tree_ : Tree
    """

    def __init__(
        self,
        *,
        n_shapelets=1,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        force_dim=None,
        criterion="mse",
        random_state=None,
    ):
        """Construct a extra shapelet tree regressor

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of the tree. If `None` the tree is expanded until all
            leaves are pure or until all leaves contain less than `min_samples_split`
            samples

        min_samples_split : int, optional
            The minimum number of samples to split an internal node

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf

        criterion : {"mse"}, optional
            The criterion used to evaluate the utility of a split

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value

        n_shapelets : int, optional
            The number of shapelets to sample at each node.

        min_shapelet_size : float, optional
            The minimum length of a sampled shapelet expressed as a fraction, computed
            as `min(ceil(X.shape[-1] * min_shapelet_size), 2)`.

        max_shapelet_size : float, optional
            The maximum length of a sampled shapelet, expressed as a fraction, computed
            as `ceil(X.shape[-1] * max_shapelet_size)`.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Distance metric used to identify the best shapelet.

        metric_params : dict, optional
            Parameters for the distance measure

        force_dim : int, optional
            Force the number of dimensions.

            - If int, force_dim reshapes the input to shape (n_samples, force_dim, -1)
              for interoperability with `sklearn`.

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator;
            - If `RandomState` instance, `random_state` is the random number generator;
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super(ExtraShapeletTreeRegressor, self).__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            n_shapelets=n_shapelets,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            force_dim=force_dim,
            random_state=random_state,
        )

    def _get_tree_builder(
        self, x, y, sample_weights, feature_engineer, random_state, max_depth
    ):
        if self.criterion not in CLF_CRITERION:
            raise ValueError("criterion (%s) is not supported" % self.criterion)

        criterion = REG_CRITERION[self.criterion](y)
        tree = Tree(feature_engineer, 1)
        return TreeBuilder(
            x,
            sample_weights,
            feature_engineer,
            criterion,
            tree,
            random_state,
            max_depth=max_depth,
            min_sample_split=self.min_samples_split,
            min_sample_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )


class ShapeletTreeClassifier(ClassifierMixin, ClassifierTreeMixin, BaseShapeletTree):
    """A shapelet tree classifier.

    Attributes
    ----------

    tree_ : Tree
        The tree data structure used internally

    classes_ : ndarray of shape (n_classes,)
        The class labels

    n_classes_ : int
        The number of class labels

    See Also
    --------
    ShapeletTreeRegressor : A shapelet tree regressor.
    ExtraShapeletTreeClassifier : An extra random shapelet tree classifier.
    """

    def __init__(
        self,
        *,
        n_shapelets=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        force_dim=None,
        class_weight=None,
        random_state=None,
    ):
        """Construct a shapelet tree classifier

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of the tree. If `None` the tree is expanded until all
            leaves are pure or until all leaves contain less than `min_samples_split`
            samples

        min_samples_split : int, optional
            The minimum number of samples to split an internal node

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf

        criterion : {"entropy", "gini"}, optional
            The criterion used to evaluate the utility of a split

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value

        n_shapelets : int, optional
            The number of shapelets to sample at each node.

        min_shapelet_size : float, optional
            The minimum length of a sampled shapelet expressed as a fraction, computed
            as `min(ceil(X.shape[-1] * min_shapelet_size), 2)`.

        max_shapelet_size : float, optional
            The maximum length of a sampled shapelet, expressed as a fraction, computed
            as `ceil(X.shape[-1] * max_shapelet_size)`.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Distance metric used to identify the best shapelet.

        metric_params : dict, optional
            Parameters for the distance measure

        force_dim : int, optional
            Force the number of dimensions.

            - If int, force_dim reshapes the input to shape (n_samples, force_dim, -1)
              for interoperability with `sklearn`.

        class_weight : dict or "balanced", optional
            Weights associated with the labels

            - if dict, weights on the form {label: weight}
            - if "balanced" each class weight inversely proportional to the class
              frequency
            - if None, each class has equal weight

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator;
            - If `RandomState` instance, `random_state` is the random number generator;
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super(ShapeletTreeClassifier, self).__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            n_shapelets=n_shapelets,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            force_dim=force_dim,
            random_state=random_state,
        )
        self.n_classes_ = None
        self.criterion = criterion
        self.class_weight = class_weight


class ExtraShapeletTreeClassifier(ShapeletTreeClassifier):
    """An extra shapelet tree classifier.

    Extra shapelet trees are constructed by sampling a distance threshold
    uniformly in the range [min(dist), max(dist)].

    Attributes
    ----------

    tree_ : Tree
    """

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_samples_split=2,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        force_dim=None,
        class_weight=None,
        random_state=None,
    ):
        """Construct a extra shapelet tree regressor

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of the tree. If `None` the tree is expanded until all
            leaves are pure or until all leaves contain less than `min_samples_split`
            samples

        min_samples_split : int, optional
            The minimum number of samples to split an internal node

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf

        criterion : {"entropy", "gini"}, optional
            The criterion used to evaluate the utility of a split

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value

        min_shapelet_size : float, optional
            The minimum length of a sampled shapelet expressed as a fraction, computed
            as `min(ceil(X.shape[-1] * min_shapelet_size), 2)`.

        max_shapelet_size : float, optional
            The maximum length of a sampled shapelet, expressed as a fraction, computed
            as `ceil(X.shape[-1] * max_shapelet_size)`.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Distance metric used to identify the best shapelet.

        metric_params : dict, optional
            Parameters for the distance measure

        force_dim : int, optional
            Force the number of dimensions.

            - If int, force_dim reshapes the input to shape (n_samples, force_dim, -1)
              for interoperability with `sklearn`.

        class_weight : dict or "balanced", optional
            Weights associated with the labels

            - if dict, weights on the form {label: weight}
            - if "balanced" each class weight inversely proportional to the class
              frequency
            - if None, each class has equal weight

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator;
            - If `RandomState` instance, `random_state` is the random number generator;
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super(ShapeletTreeClassifier, self).__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            n_shapelets=1,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            force_dim=force_dim,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.n_classes_ = None

    def _get_tree_builder(
        self, x, y, sample_weights, feature_engineer, random_state, max_depth
    ):
        if self.criterion not in CLF_CRITERION:
            raise ValueError("criterion (%s) is not supported" % self.criterion)

        criterion = CLF_CRITERION[self.criterion](y, self.n_classes_)
        tree = Tree(feature_engineer, self.n_classes_)
        return ExtraTreeBuilder(
            x,
            sample_weights,
            feature_engineer,
            criterion,
            tree,
            random_state,
            max_depth=max_depth,
            min_sample_split=self.min_samples_split,
            min_sample_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )


class BaseRocketTree(BaseTree, metaclass=ABCMeta):
    def __init__(
        self,
        n_kernels=10,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="entropy",
        sampling="auto",
        sampling_params=None,
        kernel_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        force_dim=None,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_kernels : int, optional
            The number of kernels to inspect at each node.

        max_depth : int, optional
            The maxium depth.

        min_samples_split : int, optional
            The minimum number of samples required to split.

        force_dim : int, optional
            Force reshaping of input data.

        random_state : int or RandomState, optional
            The psudo-random number generator.
        """
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            force_dim=force_dim,
        )
        self.n_kernels = n_kernels
        self.criterion = criterion
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernels_size = kernel_size
        self.bias_prob = bias_prob
        self.normalize_prob = normalize_prob
        self.padding_prob = padding_prob
        self.random_state = random_state

    def _get_feature_engineer(self):
        if self.kernel_size is None:
            kernel_size = [7, 11, 13]
        elif isinstance(self.kernel_size, tuple) and len(self.kernel_size) == 2:
            min_size, max_size = self.kernel_size
            if min_size < 0 or min_size > max_size:
                raise ValueError(
                    "`min_size` {0} <= 0 or {0} > {1}".format(min_size, max_size)
                )
            if max_size > 1:
                raise ValueError("`max_size` {0} > 1".format(max_size))
            max_size = int(self.n_timestep_ * max_size)
            min_size = int(self.n_timestep_ * min_size)
            if min_size < 2:
                min_size = 2
            kernel_size = np.arange(min_size, max_size)
        else:
            kernel_size = self.kernel_size

        if self.sampling in _SAMPLING_METHOD:
            sampling_params = (
                {} if self.sampling_params is None else self.sampling_params
            )
            weight_sampler = _SAMPLING_METHOD[self.sampling](**sampling_params)
        else:
            raise ValueError("sampling (%r) is not supported." % self.sampling)
        return RocketFeatureEngineer(
            int(self.n_kernels),
            weight_sampler,
            np.array(kernel_size, dtype=int),
            float(self.bias_prob),
            float(self.padding_prob),
            float(self.normalize_prob),
        )


class RocketTreeRegressor(RegressorMixin, RegressorTreeMixin, BaseRocketTree):
    def __init__(
        self,
        n_kernels=10,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="mse",
        sampling="auto",
        sampling_params=None,
        kernel_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        force_dim=None,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_kernels : int, optional
            The number of kernels to inspect at each node.

        max_depth : int, optional
            The maxium depth.

        min_samples_split : int, optional
            The minimum number of samples required to split.

        force_dim : int, optional
            Force reshaping of input data.

        random_state : int or RandomState, optional
            The psudo-random number generator.
        """
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            force_dim=force_dim,
        )
        self.n_kernels = n_kernels
        self.criterion = criterion
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.bias_prob = bias_prob
        self.normalize_prob = normalize_prob
        self.padding_prob = padding_prob
        self.random_state = random_state


class RocketTreeClassifier(ClassifierMixin, ClassifierTreeMixin, BaseRocketTree):
    def __init__(
        self,
        n_kernels=10,
        *,
        max_depth=None,
        min_samples_split=2,
        min_sample_leaf=1,
        min_impurity_decrease=0.0,
        criterion="entropy",
        sampling="auto",
        sampling_params=None,
        kernel_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        force_dim=None,
        class_weight=None,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_kernels : int, optional
            The number of kernels to inspect at each node.

        max_depth : int, optional
            The maxium depth.

        min_samples_split : int, optional
            The minimum number of samples required to split.

        force_dim : int, optional
            Force reshaping of input data.

        random_state : int or RandomState, optional
            The psudo-random number generator.
        """
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_sample_leaf=min_sample_leaf,
            min_impurity_decrease=min_impurity_decrease,
            force_dim=force_dim,
        )
        self.n_kernels = n_kernels
        self.criterion = criterion
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.bias_prob = bias_prob
        self.normalize_prob = normalize_prob
        self.padding_prob = padding_prob
        self.random_state = random_state
        self.class_weight = class_weight
