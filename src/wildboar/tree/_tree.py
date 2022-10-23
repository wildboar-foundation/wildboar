# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers
import sys
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.utils import check_scalar

from ..distance import _DISTANCE_MEASURE, _SUBSEQUENCE_DISTANCE_MEASURE
from ..transform._cpivot import PivotFeatureEngineer
from ..transform._crocket import RocketFeatureEngineer
from ..transform._cshapelet import RandomShapeletFeatureEngineer
from ..transform._interval import (
    _SUMMARIZER,
    IntervalFeatureEngineer,
    PyFuncSummarizer,
    RandomFixedIntervalFeatureEngineer,
    RandomIntervalFeatureEngineer,
)
from ..transform._rocket import _SAMPLING_METHOD
from ..tree._ctree import (
    DynamicTreeFeatureEngineer,
    EntropyCriterion,
    ExtraTreeBuilder,
    GiniCriterion,
    MSECriterion,
    Tree,
    TreeBuilder,
    TreeFeatureEngineer,
)
from ..utils.data import check_dataset
from ..utils.validation import check_option
from .base import BaseTree, TreeClassifierMixin, TreeRegressorMixin

CLF_CRITERION = {"gini": GiniCriterion, "entropy": EntropyCriterion}
REG_CRITERION = {"mse": MSECriterion, "squared_error": MSECriterion}


class BaseFeatureTree(BaseTree, metaclass=ABCMeta):
    """Base class for trees using feature engineering."""

    @abstractmethod
    def _get_feature_engineer(self, n_samples):
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

    def _wrap_feature_engineer(self, feature_engineer):
        return TreeFeatureEngineer(feature_engineer)

    def _fit(self, x, y, sample_weights, random_state):
        feature_engineer = self._wrap_feature_engineer(
            self._get_feature_engineer(self.n_timesteps_in_)
        )
        x = check_dataset(x)
        tree_builder = self._get_tree_builder(
            x,
            y,
            sample_weights,
            feature_engineer,
            random_state,
            check_scalar(
                sys.getrecursionlimit() if self.max_depth is None else self.max_depth,
                "max_depth",
                numbers.Integral,
                min_val=1,
                max_val=sys.getrecursionlimit(),
            ),
        )
        tree_builder.build_tree()
        self.tree_ = tree_builder.tree_


class FeatureTreeRegressorMixin(TreeRegressorMixin):
    def _get_tree_builder(
        self, x, y, sample_weights, feature_engineer, random_state, max_depth
    ):
        # TODO(1.2): remove
        if self.criterion == "mse":
            warnings.warn(
                "Criterion 'mse' was deprecated in v1.1 and will be "
                "removed in version 1.2. Use criterion='squared_error' "
                "which is equivalent.",
                FutureWarning,
            )
        Criterion = check_option(REG_CRITERION, self.criterion, "criterion")
        return TreeBuilder(
            x,
            sample_weights,
            feature_engineer,
            Criterion(y),
            Tree(feature_engineer, 1),
            random_state,
            max_depth=max_depth,
            min_sample_split=check_scalar(
                self.min_samples_split, "min_samples_split", numbers.Real, min_val=2
            ),
            min_sample_leaf=check_scalar(
                self.min_samples_leaf, "min_samples_leaf", numbers.Real, min_val=1
            ),
            min_impurity_decrease=check_scalar(
                self.min_impurity_decrease,
                "min_impurity_decrease",
                numbers.Real,
                min_val=0,
            ),
        )


class FeatureTreeClassifierMixin(TreeClassifierMixin):
    def _get_tree_builder(
        self, x, y, sample_weights, feature_engineer, random_state, max_depth
    ):
        Criterion = check_option(CLF_CRITERION, self.criterion, "criterion")
        return TreeBuilder(
            x,
            sample_weights,
            feature_engineer,
            Criterion(y, self.n_classes_),
            Tree(feature_engineer, self.n_classes_),
            random_state,
            max_depth=max_depth,
            min_sample_split=check_scalar(
                self.min_samples_split, "min_samples_split", numbers.Real, min_val=2
            ),
            min_sample_leaf=check_scalar(
                self.min_samples_leaf, "min_samples_leaf", numbers.Real, min_val=1
            ),
            min_impurity_decrease=check_scalar(
                self.min_impurity_decrease,
                "min_impurity_decrease",
                numbers.Real,
                min_val=0,
            ),
        )


class DynamicTreeMixin:
    def _wrap_feature_engineer(self, feature_engineer):
        if hasattr(self, "alpha") and self.alpha is not None:
            if self.alpha == 0.0:
                raise ValueError("alpha == 0.0, must be != 0")

            return DynamicTreeFeatureEngineer(feature_engineer, self.alpha)

        return TreeFeatureEngineer(feature_engineer)


class BaseShapeletTree(BaseFeatureTree):
    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        n_shapelets="warn",
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        random_state=None,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.random_state = random_state
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params

    def _get_feature_engineer(self, n_samples):
        check_scalar(
            self.max_shapelet_size,
            "max_shapelet_size",
            numbers.Real,
            min_val=self.min_shapelet_size,
            max_val=1.0,
        )
        check_scalar(
            self.min_shapelet_size,
            "min_shapelet_size",
            numbers.Real,
            min_val=0.0,
            max_val=self.max_shapelet_size,
        )

        max_shapelet_size = math.ceil(self.n_timesteps_in_ * self.max_shapelet_size)
        min_shapelet_size = math.ceil(self.n_timesteps_in_ * self.min_shapelet_size)
        if min_shapelet_size < 2:
            # NOTE: To ensure that the same random_seed generates the same shapelets
            # in future versions we keep the limit of 2 timesteps for a shapelet as long
            # as the time series is at least 2 timesteps. Otherwise we fall back to 1
            # timestep.
            #
            # TODO(1.2): consider breaking backwards compatibility and always limit to
            #            1 timestep.
            if self.n_timesteps_in_ < 2:
                min_shapelet_size = 1
            else:
                min_shapelet_size = 2

        # TODO(1.2): change the default value
        if self.n_shapelets == "warn":
            warnings.warn(
                "The default value of n_shapelets will change from 10 to 'log2' in 1.2",
                FutureWarning,
            )
            n_shapelets = 10
        elif isinstance(self.n_shapelets, str) or callable(self.n_shapelets):
            if min_shapelet_size < max_shapelet_size:
                possible_shapelets = sum(
                    self.n_timesteps_in_ - curr_len + 1
                    for curr_len in range(min_shapelet_size, max_shapelet_size)
                )
            else:
                possible_shapelets = self.n_timesteps_in_ - min_shapelet_size + 1

            if self.n_shapelets == "log2":
                n_shapelets = int(np.log2(possible_shapelets))
            elif self.n_shapelets == "sqrt":
                n_shapelets = int(np.sqrt(possible_shapelets))
            elif callable(self.n_shapelets):
                n_shapelets = int(self.n_shapelets(possible_shapelets))
            else:
                raise ValueError(
                    "n_shapelets must be 'log2', 'sqrt' or callable, got %r"
                    % self.n_shapelets
                )
        else:
            n_shapelets = check_scalar(
                self.n_shapelets, "n_shapelets", numbers.Integral, min_val=1
            )

        DistanceMeasure = check_option(
            _SUBSEQUENCE_DISTANCE_MEASURE, self.metric, "metric"
        )
        metric_params = self.metric_params if self.metric_params is not None else {}
        return RandomShapeletFeatureEngineer(
            DistanceMeasure(**metric_params),
            min_shapelet_size,
            max_shapelet_size,
            max(1, n_shapelets),
        )


class ShapeletTreeRegressor(
    DynamicTreeMixin, FeatureTreeRegressorMixin, BaseShapeletTree
):
    """A shapelet tree regressor.

    Attributes
    ----------

    tree_ : Tree
        The internal tree representation

    """

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        n_shapelets="warn",
        min_shapelet_size=0,
        max_shapelet_size=1,
        alpha=None,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
        random_state=None,
    ):
        """
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

        criterion : {"squared_error"}, optional
            The criterion used to evaluate the utility of a split

            .. deprecated:: 1.0
                Criterion "mse" was deprecated in v1.1 and will be removed in
                version 1.2. Use `criterion="squared_error"` which is equivalent.

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

        alpha : float, optional
            Dynamically decrease the number of sampled shapelets at each node according
            to the current depth.

            .. math:: w = 1 - e^{-|alpha| * depth}

            - if :math:`alpha < 0`, the number of sampled shapelets decrease from
              ``n_shapelets`` towards 1 with increased depth.

              .. math:: n_shapelets * (1 - w)

            - if :math:`alpha > 0`, the number of sampled shapelets increase from ``1``
              towards ``n_shapelets`` with increased depth.

              .. math:: n_shapelets * w

            - if ``None``, the number of sampled shapelets are the same independeth of
              depth.

        metric : str, optional
            Distance metric used to identify the best shapelet.

            See ``distance._SUBSEQUENCE_DISTANCE_MEASURE.keys()`` for a list of
            supported metrics.

        metric_params : dict, optional
            Parameters for the distance measure.

            Read more about the parameters in the
            :ref:`User guide <list_of_subsequence_metrics>`.

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
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
            random_state=random_state,
        )
        self.criterion = criterion
        self.alpha = alpha


class ExtraShapeletTreeRegressor(ShapeletTreeRegressor):
    """An extra shapelet tree regressor.

    Extra shapelet trees are constructed by sampling a distance threshold
    uniformly in the range [min(dist), max(dist)].

    Attributes
    ----------

    tree_ : Tree
        The internal tree representation

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
        criterion="squared_error",
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_shapelets : int, optional
            The number of shapelets to sample at each node.

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

            .. deprecated:: 1.0
                Criterion "mse" was deprecated in v1.1 and will be removed in
                version 1.2. Use `criterion="squared_error"` which is equivalent.

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
            random_state=random_state,
        )

    def _get_tree_builder(
        self, x, y, sample_weights, feature_engineer, random_state, max_depth
    ):
        # TODO(1.2): remove
        if self.criterion == "mse":
            warnings.warn(
                "Criterion 'mse' was deprecated in v1.1 and will be "
                "removed in version 1.2. Use criterion='squared_error' "
                "which is equivalent.",
                FutureWarning,
            )
        Criterion = check_option(REG_CRITERION, self.criterion, "criterion")
        return ExtraTreeBuilder(
            x,
            sample_weights,
            feature_engineer,
            Criterion(y),
            Tree(feature_engineer, 1),
            random_state,
            max_depth=max_depth,
            min_sample_split=check_scalar(
                self.min_samples_split, "min_samples_split", numbers.Real, min_val=2
            ),
            min_sample_leaf=check_scalar(
                self.min_samples_leaf, "min_samples_leaf", numbers.Real, min_val=1
            ),
            min_impurity_decrease=check_scalar(
                self.min_impurity_decrease,
                "min_impurity_decrease",
                numbers.Real,
                min_val=0,
            ),
        )


class ShapeletTreeClassifier(
    DynamicTreeMixin, FeatureTreeClassifierMixin, BaseShapeletTree
):
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
        n_shapelets="warn",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        alpha=None,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        class_weight=None,
        random_state=None,
    ):
        """

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
            as ``min(ceil(X.shape[-1] * min_shapelet_size), 2)``.

        max_shapelet_size : float, optional
            The maximum length of a sampled shapelet, expressed as a fraction, computed
            as ``ceil(X.shape[-1] * max_shapelet_size)``.

        alpha : float, optional
            Dynamically decrease the number of sampled shapelets at each node according
            to the current depth.

            .. math:`w = 1 - e^{-|alpha| * depth})`

            - if :math:`alpha < 0`, the number of sampled shapelets decrease from
              ``n_shapelets`` towards 1 with increased depth.

              .. math:`n_shapelets * (1 - w)`

            - if :math:`alpha > 0`, the number of sampled shapelets increase from ``1``
              towards ``n_shapelets`` with increased depth.

              .. math:`n_shapelets * w`

            - if ``None``, the number of sampled shapelets are the same independeth of
              depth.

        metric : {"euclidean", "scaled_euclidean", "dtw", "scaled_dtw"}, optional
            Distance metric used to identify the best shapelet.

        metric_params : dict, optional
            Parameters for the distance measure

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
            random_state=random_state,
        )
        self.criterion = criterion
        self.class_weight = class_weight
        self.alpha = alpha


class ExtraShapeletTreeClassifier(ShapeletTreeClassifier):
    """An extra shapelet tree classifier.

    Extra shapelet trees are constructed by sampling a distance threshold
    uniformly in the range ``[min(dist), max(dist)]``.

    Attributes
    ----------
    tree_ : Tree
        The tree representation

    """

    def __init__(
        self,
        *,
        n_shapelets=1,
        max_depth=None,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_samples_split=2,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        class_weight=None,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_shapelets : int, optional
            The number of shapelets to sample at each node.

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
            as ``min(ceil(X.shape[-1] * min_shapelet_size), 2)``.

        max_shapelet_size : float, optional
            The maximum length of a sampled shapelet, expressed as a fraction, computed
            as ``ceil(X.shape[-1] * max_shapelet_size)``.

        metric : {"euclidean", "scaled_euclidean", "dtw", "scaled_dtw"}, optional
            Distance metric used to identify the best shapelet.

        metric_params : dict, optional
            Parameters for the distance measure

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
            n_shapelets=n_shapelets,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            class_weight=class_weight,
            random_state=random_state,
        )

    def _get_tree_builder(
        self, x, y, sample_weights, feature_engineer, random_state, max_depth
    ):
        Criterion = check_option(CLF_CRITERION, self.criterion, "criterion")
        return ExtraTreeBuilder(
            x,
            sample_weights,
            feature_engineer,
            Criterion(y, self.n_classes_),
            Tree(feature_engineer, self.n_classes_),
            random_state,
            max_depth=max_depth,
            min_sample_split=check_scalar(
                self.min_samples_split, "min_samples_split", numbers.Real, min_val=2
            ),
            min_sample_leaf=check_scalar(
                self.min_samples_leaf, "min_samples_leaf", numbers.Real, min_val=1
            ),
            min_impurity_decrease=check_scalar(
                self.min_impurity_decrease,
                "min_impurity_decrease",
                numbers.Real,
                min_val=0,
            ),
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
        sampling="normal",
        sampling_params=None,
        kernel_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        random_state=None,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
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

    def _get_feature_engineer(self, n_samples):
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
            max_size = int(self.n_timesteps_in_ * max_size)
            min_size = int(self.n_timesteps_in_ * min_size)
            if min_size < 2:
                if self.n_timesteps_in_ < 2:
                    min_size = 1
                else:
                    min_size = 2
            kernel_size = np.arange(min_size, max_size)
        else:
            kernel_size = self.kernel_size

        WeightSampler = check_option(_SAMPLING_METHOD, self.sampling, "sampling")
        sampling_params = {} if self.sampling_params is None else self.sampling_params
        return RocketFeatureEngineer(
            check_scalar(self.n_kernels, "n_kernels", numbers.Integral, min_val=1),
            WeightSampler(**sampling_params),
            np.array(kernel_size, dtype=int),
            check_scalar(
                self.bias_prob, "bias_prob", numbers.Real, min_val=0, max_val=1
            ),
            check_scalar(
                self.padding_prob, "padding_prob", numbers.Real, min_val=0, max_val=1
            ),
            check_scalar(
                self.normalize_prob,
                "normalize_prob",
                numbers.Real,
                min_val=0,
                max_val=1,
            ),
        )


class RocketTreeRegressor(FeatureTreeRegressorMixin, BaseRocketTree):
    """A tree regressor that uses random convolutions as features.

    Attributes
    ----------

    tree_ : Tree
        The internal tree representation.

    """

    def __init__(
        self,
        n_kernels=10,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="squared_error",
        sampling="normal",
        sampling_params=None,
        kernel_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_kernels : int, optional
            The number of kernels to sample at each node.

        max_depth : int, optional
            The maximum depth of the tree. If `None` the tree is expanded until all
            leaves are pure or until all leaves contain less than `min_samples_split`
            samples.

        min_samples_split : int, optional
            The minimum number of samples to split an internal node.

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf.

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value.

        criterion : {"entropy", "gini"}, optional
            The criterion used to evaluate the utility of a split.

        sampling : {"normal", "uniform", "shapelet"}, optional
            The sampling of convolutional filters.

            - if "normal", sample filter according to a normal distribution with
              ``mean`` and ``scale``.

            - if "uniform", sample filter according to a uniform distribution with
              ``lower`` and ``upper``.

            - if "shapelet", sample filters as subsequences in the training data.

        sampling_params : dict, optional
            The parameters for the sampling.

            - if "normal", ``{"mean": float, "scale": float}``, defaults to
               ``{"mean": 0, "scale": 1}``.

            - if "uniform", ``{"lower": float, "upper": float}``, defaults to
               ``{"lower": -1, "upper": 1}``.

        kernel_size : (min_size, max_size) or array-like, optional
            The kernel size.

            - if (min_size, max_size), all kernel sizes between
              ``min_size * n_timestep`` and ``max_size * n_timestep``

            - if array-like, all defined kernel sizes.

        bias_prob : float, optional
            The probability of using a bias term.

        normalize_prob : float, optional
            The probability of performing normalization.

        padding_prob : float, optional
            The probability of padding with zeros.

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
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
    """A tree classifier that uses random convolutions as features.

    Attributes
    ----------

    tree_ : Tree
        The internal tree representation.

    """

    def __init__(
        self,
        n_kernels=10,
        *,
        max_depth=None,
        min_samples_split=2,
        min_sample_leaf=1,
        min_impurity_decrease=0.0,
        criterion="entropy",
        sampling="normal",
        sampling_params=None,
        kernel_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        class_weight=None,
        random_state=None,
    ):
        """
        Parameters
        ----------

        n_kernels : int, optional
            The number of kernels to sample at each node.

        max_depth : int, optional
            The maximum depth of the tree. If `None` the tree is expanded until all
            leaves are pure or until all leaves contain less than `min_samples_split`
            samples.

        min_samples_split : int, optional
            The minimum number of samples to split an internal node.

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf.

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value.

        criterion : {"entropy", "gini"}, optional
            The criterion used to evaluate the utility of a split.

        sampling : {"normal", "uniform", "shapelet"}, optional
            The sampling of convolutional filters.

            - if "normal", sample filter according to a normal distribution with
              ``mean`` and ``scale``.

            - if "uniform", sample filter according to a uniform distribution with
              ``lower`` and ``upper``.

            - if "shapelet", sample filters as subsequences in the training data.

        sampling_params : dict, optional
            The parameters for the sampling.

            - if "normal", ``{"mean": float, "scale": float}``, defaults to
               ``{"mean": 0, "scale": 1}``.

            - if "uniform", ``{"lower": float, "upper": float}``, defaults to
               ``{"lower": -1, "upper": 1}``.

        kernel_size : (min_size, max_size) or array-like, optional
            The kernel size.

            - if (min_size, max_size), all kernel sizes between
              ``min_size * n_timestep`` and ``max_size * n_timestep``

            - if array-like, all defined kernel sizes.

        bias_prob : float, optional
            The probability of using a bias term.

        normalize_prob : float, optional
            The probability of performing normalization.

        padding_prob : float, optional
            The probability of padding with zeros.

        class_weight : dict or "balanced", optional
            Weights associated with the labels

            - if dict, weights on the form {label: weight}
            - if "balanced" each class weight inversely proportional to the class
              frequency
            - if None, each class has equal weight

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_sample_leaf=min_sample_leaf,
            min_impurity_decrease=min_impurity_decrease,
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
        n_intervals="sqrt",
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        intervals="fixed",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        summarizer="mean_var_slope",
        random_state=None,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.n_intervals = n_intervals
        self.intervals = intervals
        self.sample_size = sample_size
        self.min_size = min_size
        self.max_size = max_size
        self.summarizer = summarizer
        self.random_state = random_state

    def _get_feature_engineer(self, n_samples):
        if isinstance(self.summarizer, list):
            if not all(hasattr(func, "__call__") for func in self.summarizer):
                raise ValueError(
                    "summarizer must be list of callable or str, got %r"
                    % self.summarizer
                )
            summarizer = PyFuncSummarizer(self.summarizer)
        else:
            summarizer = check_option(_SUMMARIZER, self.summarizer, "summarizer")()

        if self.n_intervals == "sqrt":
            n_intervals = math.ceil(math.sqrt(self.n_timesteps_in_))
        elif self.n_intervals == "log":
            n_intervals = math.ceil(math.log2(self.n_timesteps_in_))
        elif isinstance(self.n_intervals, numbers.Integral):
            n_intervals = check_scalar(
                self.n_intervals,
                "n_intervals",
                numbers.Integral,
                min_val=1,
                max_val=self.n_timesteps_in_,
            )
        elif isinstance(self.n_intervals, numbers.Real):
            n_intervals = math.ceil(
                check_scalar(
                    self.n_intervals,
                    "n_intervals",
                    numbers.Real,
                    min_val=0,
                    max_val=1,
                    include_boundaries="right",
                )
                * self.n_timesteps_in_
            )
        else:
            raise TypeError(
                "n_intervals must be 'sqrt', 'log', float or int, got %r"
                % type(self.n_intervals).__qualname__
            )

        if self.intervals == "fixed":
            return IntervalFeatureEngineer(n_intervals, summarizer)
        elif self.intervals == "sample":
            sample_size = math.floor(
                check_scalar(
                    self.sample_size, "sample_size", numbers.Real, min_val=0, max_val=1
                )
                * n_intervals
            )
            return RandomFixedIntervalFeatureEngineer(
                n_intervals, summarizer, sample_size
            )
        elif self.intervals == "random":
            check_scalar(
                self.max_size,
                "self.max_size",
                numbers.Real,
                min_val=self.min_size,
                max_val=1.0,
                include_boundaries="right",
            )
            check_scalar(
                self.min_size,
                "self.min_size",
                numbers.Real,
                min_val=0.0,
                max_val=self.max_size,
                include_boundaries="right",
            )

            min_size = int(self.min_size * self.n_timesteps_in_)
            max_size = int(self.max_size * self.n_timesteps_in_)
            if min_size < 2:
                if self.n_timesteps_in_ < 2:
                    min_size = 1
                else:
                    min_size = 2

            return RandomIntervalFeatureEngineer(
                n_intervals, summarizer, min_size, max_size
            )
        else:
            raise ValueError(
                "intervals must be 'fixed', 'sample' or 'random', got %r"
                % self.intervals
            )


class IntervalTreeClassifier(FeatureTreeClassifierMixin, BaseIntervalTree):
    """An interval based tree classifier.

    Attributes
    ----------

    tree_ : Tree
        The internal tree structure.

    """

    def __init__(
        self,
        n_intervals="sqrt",
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
        summarizer="mean_var_slope",
        class_weight=None,
        random_state=None,
    ):
        """
        Parameters
        ----------

        n_intervals : {"log", "sqrt"}, int or float, optional
            The number of intervals to partition the time series into.

            - if "log", the number of intervals is ``log2(n_timestep)``.
            - if "sqrt", the number of intervals is ``sqrt(n_timestep)``.
            - if int, the number of intervals is ``n_intervals``.
            - if float, the number of intervals is ``n_intervals * n_timestep``, with
              ``0 < n_intervals < 1``.

        max_depth : int, optional
            The maximum depth of the tree. If `None` the tree is expanded until all
            leaves are pure or until all leaves contain less than `min_samples_split`
            samples.

        min_samples_split : int, optional
            The minimum number of samples to split an internal node.

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf.

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value.

        criterion : {"entropy", "gini"}, optional
            The criterion used to evaluate the utility of a split.

        intervals : {"fixed", "sample", "random"}, optional

            - if "fixed", `n_intervals` non-overlapping intervals.
            - if "sample", ``n_intervals * sample_size`` non-overlapping intervals.
            - if "random", `n_intervals` possibly overlapping intervals of randomly
              sampled in ``[min_size * n_timestep, max_size * n_timestep]``

        sample_size : float, optional
            The fraction of intervals to sample at each node. Ignored unless
            ``intervals="sample"``.

        min_size : float, optional
            The minmum interval size. Ignored unless ``intervals="random"``.

        max_size : float, optional
            The maximum interval size. Ignored unless ``intervals="random"``.

        summarizer : list or str, optional
            The summarization of each interval.

            - if list, a list of callables accepting a numpy array returing a float.
            - if str, a predified summarized. See
              :mod:`wildboar.transform._interval._INTERVALS.keys()` for all supported
              summarizers.

        class_weight : dict or "balanced", optional
            Weights associated with the labels

            - if dict, weights on the form {label: weight}
            - if "balanced" each class weight inversely proportional to the class
              frequency
            - if None, each class has equal weight

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(
            n_intervals=n_intervals,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            intervals=intervals,
            sample_size=sample_size,
            min_size=min_size,
            max_size=max_size,
            summarizer=summarizer,
            random_state=random_state,
        )
        self.class_weight = class_weight
        self.criterion = criterion


class IntervalTreeRegressor(FeatureTreeRegressorMixin, BaseIntervalTree):
    """An interval based tree regressor.

    Attributes
    ----------

    tree_ : Tree
        The internal tree structure.

    """

    def __init__(
        self,
        n_intervals="sqrt",
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="squared_error",
        intervals="fixed",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        summarizer="mean_var_slope",
        random_state=None,
    ):
        """
        Parameters
        ----------

        n_intervals : {"log", "sqrt"}, int or float, optional
            The number of intervals to partition the time series into.

            - if "log", the number of intervals is ``log2(n_timestep)``.
            - if "sqrt", the number of intervals is ``sqrt(n_timestep)``.
            - if int, the number of intervals is ``n_intervals``.
            - if float, the number of intervals is ``n_intervals * n_timestep``, with
              ``0 < n_intervals < 1``.

        max_depth : int, optional
            The maximum depth of the tree. If `None` the tree is expanded until all
            leaves are pure or until all leaves contain less than `min_samples_split`
            samples.

        min_samples_split : int, optional
            The minimum number of samples to split an internal node.

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf.

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value.

        criterion : {"squared_error"}, optional
            The criterion used to evaluate the utility of a split.

            .. deprecated:: 1.0
                Criterion "mse" was deprecated in v1.1 and will be removed in
                version 1.2. Use `criterion="squared_error"` which is equivalent.

        intervals : {"fixed", "sample", "random"}, optional

            - if "fixed", `n_intervals` non-overlapping intervals.
            - if "sample", ``n_intervals * sample_size`` non-overlapping intervals.
            - if "random", `n_intervals` possibly overlapping intervals of randomly
              sampled in ``[min_size * n_timestep, max_size * n_timestep]``

        sample_size : float, optional
            The fraction of intervals to sample at each node. Ignored unless
            ``intervals="sample"``.

        min_size : float, optional
            The minmum interval size. Ignored unless ``intervals="random"``.

        max_size : float, optional
            The maximum interval size. Ignored unless ``intervals="random"``.

        summarizer : list or str, optional
            The summarization of each interval.

            - if list, a list of callables accepting a numpy array returing a float.
            - if str, a predified summarized. See
              :mod:`wildboar.transform._interval._INTERVALS.keys()` for all supported
              summarizers.

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(
            n_intervals=n_intervals,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            intervals=intervals,
            sample_size=sample_size,
            min_size=min_size,
            max_size=max_size,
            summarizer=summarizer,
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
        random_state=None,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.n_pivot = n_pivot
        self.metrics = metrics
        self.random_state = random_state

    def _get_feature_engineer(self, n_samples):
        if self.n_pivot == "sqrt":
            n_pivot = math.ceil(math.sqrt(n_samples))
        elif self.n_pivot == "log":
            n_pivot = math.ceil(math.log2(n_samples))
        elif isinstance(self.n_pivot, numbers.Integral):
            n_pivot = check_scalar(
                self.n_pivot, "n_pivot", numbers.Integral, min_val=1, max_val=n_samples
            )
        elif isinstance(self.n_pivot, numbers.Real):
            n_pivot = math.ceil(
                n_samples
                * check_scalar(
                    self.n_pivot,
                    "n_pivot",
                    numbers.Real,
                    min_val=0,
                    max_val=1,
                    include_boundaries="right",
                )
            )
        else:
            raise ValueError(
                "n_pivot must be 'sqrt', 'log', int or float, got %r" % self.n_pivot
            )
        metrics = [_DISTANCE_MEASURE["dtw"](r) for r in np.linspace(0.1, 0.4, 8)]
        return PivotFeatureEngineer(
            n_pivot, [_DISTANCE_MEASURE["euclidean"]()] + metrics
        )


class PivotTreeClassifier(FeatureTreeClassifierMixin, BasePivotTree):
    """A tree classifier that uses pivot time series.


    Attributes
    ----------

    tree_ : Tree
        The internal tree representation

    """

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
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_pivot : str or int, optional
            The number of pivot time series to sample at each node.

        metrics : str, optional
            The metrics to sample from. Currently, we only support "all".

        max_depth : int, optional
            The maximum depth of the tree. If `None` the tree is expanded until all
            leaves are pure or until all leaves contain less than `min_samples_split`
            samples.

        min_samples_split : int, optional
            The minimum number of samples to split an internal node.

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf.

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value.

        criterion : {"entropy", "gini"}, optional
            The criterion used to evaluate the utility of a split.

        class_weight : dict or "balanced", optional
            Weights associated with the labels.

            - if dict, weights on the form {label: weight}.
            - if "balanced" each class weight inversely proportional to the class
              frequency.
            - if None, each class has equal weight.

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(
            n_pivot=n_pivot,
            metrics=metrics,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
        )
        self.criterion = criterion
        self.class_weight = class_weight
