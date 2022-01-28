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
import numbers
import sys
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.utils import check_random_state

from wildboar.distance import _DISTANCE_MEASURE, _SUBSEQUENCE_DISTANCE_MEASURE
from wildboar.embed._interval import (
    _SUMMARIZER,
    IntervalFeatureEngineer,
    PyFuncSummarizer,
    RandomFixedIntervalFeatureEngineer,
    RandomIntervalFeatureEngineer,
)
from wildboar.embed._pivot import PivotFeatureEngineer
from wildboar.embed._rocket import _SAMPLING_METHOD, RocketFeatureEngineer
from wildboar.embed._shapelet import RandomShapeletFeatureEngineer
from wildboar.tree._ctree import (
    EntropyCriterion,
    ExtraTreeBuilder,
    GiniCriterion,
    MSECriterion,
    Tree,
    TreeBuilder,
)
from wildboar.utils import check_dataset

from .base import BaseTree, TreeClassifierMixin, TreeRegressorMixin

CLF_CRITERION = {"gini": GiniCriterion, "entropy": EntropyCriterion}
REG_CRITERION = {"mse": MSECriterion}


class BaseFeatureTree(BaseTree, metaclass=ABCMeta):
    """Base class for trees using feature engineering."""

    @abstractmethod
    def _get_feature_engineer(self, n_samples, n_timestep):
        """Get the feature engineer

        Returns
        -------

        FeatureEngineer : the feature engineer
        """

    @abstractmethod
    def _get_tree_builder(
        self, x, y, sample_weights, feature_engineer, random_state, max_depth
    ):
        """Get the tree builder

        Returns
        -------

        TreeBuilder : the tree builder
        """

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

        feature_engineer = self._get_feature_engineer(x.shape[0], x.shape[-1])
        x = check_dataset(x)
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


class FeatureTreeRegressorMixin(TreeRegressorMixin):
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


class FeatureTreeClassifierMixin(TreeClassifierMixin):
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


class BaseShapeletTree(BaseFeatureTree):
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

    def _get_feature_engineer(self, n_samples, n_timestep):
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

        max_shapelet_size = int(n_timestep * self.max_shapelet_size)
        min_shapelet_size = int(n_timestep * self.min_shapelet_size)
        if min_shapelet_size < 2:
            min_shapelet_size = 2

        distance_measure = _SUBSEQUENCE_DISTANCE_MEASURE.get(self.metric, None)
        if distance_measure is None:
            raise ValueError("invalid distance measure (%r)" % self.metric)
        metric_params = self.metric_params or {}

        return RandomShapeletFeatureEngineer(
            distance_measure(**metric_params),
            min_shapelet_size,
            max_shapelet_size,
            self.n_shapelets,
        )


class ShapeletTreeRegressor(FeatureTreeRegressorMixin, BaseShapeletTree):
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


class ShapeletTreeClassifier(FeatureTreeClassifierMixin, BaseShapeletTree):
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
        super(ExtraShapeletTreeClassifier, self).__init__(
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


class BaseRocketTree(BaseFeatureTree, metaclass=ABCMeta):
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

    def _get_feature_engineer(self, n_samples, n_timestep):
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
            max_size = int(n_timestep * max_size)
            min_size = int(n_timestep * min_size)
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


class RocketTreeRegressor(FeatureTreeRegressorMixin, BaseRocketTree):
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


class RocketTreeClassifier(FeatureTreeClassifierMixin, BaseRocketTree):
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


class BaseIntervalTree(BaseFeatureTree, metaclass=ABCMeta):
    def __init__(
        self,
        n_interval="sqrt",
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        intervals="fixed",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        summarizer="auto",
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
        self.n_interval = n_interval
        self.intervals = intervals
        self.sample_size = sample_size
        self.min_size = min_size
        self.max_size = max_size
        self.summarizer = summarizer
        self.random_state = random_state

    def _get_feature_engineer(self, n_samples, n_timestep):
        if isinstance(self.summarizer, list):
            if not all(hasattr(func, "__call__") for func in self.summarizer):
                raise ValueError("summarizer (%r) is not supported")
            summarizer = PyFuncSummarizer(self.summarizer)
        else:
            summarizer = _SUMMARIZER.get(self.summarizer)()
            if summarizer is None:
                raise ValueError("summarizer (%r) is not supported." % self.summarizer)

        if self.n_interval == "sqrt":
            n_interval = math.ceil(math.sqrt(n_timestep))
        elif self.n_interval == "log":
            n_interval = math.ceil(math.log2(n_timestep))
        elif isinstance(self.n_interval, numbers.Integral):
            n_interval = self.n_interval
        elif isinstance(self.n_interval, numbers.Real):
            if not 0.0 < self.n_interval < 1.0:
                raise ValueError("n_interval must be between 0.0 and 1.0")
            n_interval = math.floor(self.n_interval * n_timestep)
            # TODO: ensure that no interval is smaller than 2
        else:
            raise ValueError("n_interval (%r) is not supported" % self.n_interval)

        if self.intervals == "fixed":
            return IntervalFeatureEngineer(n_interval, summarizer)
        elif self.intervals == "sample":
            if not 0.0 < self.sample_size < 1.0:
                raise ValueError("sample_size must be between 0.0 and 1.0")

            sample_size = math.floor(n_interval * self.sample_size)
            return RandomFixedIntervalFeatureEngineer(
                n_interval, summarizer, sample_size
            )
        elif self.intervals == "random":
            if not 0.0 <= self.min_size < self.max_size:
                raise ValueError("min_size must be between 0.0 and max_size")
            if not self.min_size < self.max_size <= 1.0:
                raise ValueError("max_size must be between min_size and 1.0")

            min_size = int(self.min_size * self.n_timestep_)
            max_size = int(self.max_size * self.n_timestep_)
            if min_size < 2:
                min_size = 2

            return RandomIntervalFeatureEngineer(
                n_interval, summarizer, min_size, max_size
            )
        else:
            raise ValueError("intervals (%r) is unsupported." % self.intervals)


class IntervalTreeClassifier(FeatureTreeClassifierMixin, BaseIntervalTree):
    def __init__(
        self,
        n_interval="sqrt",
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="entropy",
        intervals="fixed",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        summarizer="auto",
        force_dim=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            n_interval=n_interval,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            intervals=intervals,
            sample_size=sample_size,
            min_size=min_size,
            max_size=max_size,
            summarizer=summarizer,
            force_dim=force_dim,
            random_state=random_state,
        )
        self.class_weight = class_weight
        self.criterion = criterion


class IntervalTreeRegressor(FeatureTreeRegressorMixin, BaseIntervalTree):
    def __init__(
        self,
        n_interval="sqrt",
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="entropy",
        intervals="fixed",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        summarizer="auto",
        force_dim=None,
        random_state=None,
    ):
        super().__init__(
            n_interval=n_interval,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            intervals=intervals,
            sample_size=sample_size,
            min_size=min_size,
            max_size=max_size,
            summarizer=summarizer,
            force_dim=force_dim,
            random_state=random_state,
        )
        self.criterion = criterion


class BasePivotTree(BaseFeatureTree, metaclass=ABCMeta):
    def __init__(
        self,
        n_pivot="sqrt",
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        metrics="all",
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
        self.n_pivot = n_pivot
        self.metrics = metrics
        self.random_state = random_state

    def _get_feature_engineer(self, n_samples, n_timestep):
        if isinstance(self.n_pivot, str):
            if self.n_pivot == "sqrt":
                n_pivot = math.ceil(math.sqrt(n_samples))
            elif self.n_pivot == "log":
                n_pivot = math.ceil(math.log2(n_samples))
            else:
                raise ValueError("invalid n_pivot (%s)" % self.n_pivot)
        elif isinstance(self.n_pivot, numbers.Integral):
            n_pivot = self.n_pivot
        elif isinstance(self.n_pivot, numbers.Real):
            if not 0 < self.n_pivot <= 1.0:
                raise ValueError(
                    "invalid n_pivots, got %d expected in range [0, 1]" % self.n_pivot
                )
            n_pivot = math.ceil(self.n_pivot * n_samples)
        else:
            raise ValueError("invalid n_pivot (%r)" % self.n_pivot)
        metrics = [_DISTANCE_MEASURE["dtw"](r) for r in np.linspace(0.1, 0.4, 8)]
        return PivotFeatureEngineer(
            n_pivot, [_DISTANCE_MEASURE["euclidean"]()] + metrics
        )


class PivotTreeClassifier(FeatureTreeClassifierMixin, BasePivotTree):
    def __init__(
        self,
        n_pivot="sqrt",
        *,
        metrics="all",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="entropy",
        class_weight=None,
        force_dim=None,
        random_state=None,
    ):
        super().__init__(
            n_pivot=n_pivot,
            metrics=metrics,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            force_dim=force_dim,
            random_state=random_state,
        )
        self.criterion = criterion
        self.class_weight = class_weight
