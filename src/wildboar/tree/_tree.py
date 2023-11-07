# Authors: Isak Samsten
# License: BSD 3 clause


from sklearn.utils._param_validation import StrOptions

from ..transform._interval import IntervalMixin
from ..transform._pivot import PivotMixin
from ..transform._rocket import RocketMixin
from ..transform._shapelet import DilatedShapeletMixin, ShapeletMixin
from ..tree._ctree import (
    DynamicTreeAttributeGenerator,
    EntropyCriterion,
    ExtraTreeBuilder,
    GiniCriterion,
    MSECriterion,
    Tree,
    TreeAttributeGenerator,
    TreeBuilder,
)
from ..utils.validation import check_option
from ._base import BaseTree, BaseTreeClassifier, BaseTreeRegressor

CLF_CRITERION = {"gini": GiniCriterion, "entropy": EntropyCriterion}
REG_CRITERION = {"squared_error": MSECriterion}


class FeatureTreeMixin:
    def _wrap_generator(self, generator):
        return TreeAttributeGenerator(generator)

    def _fit(self, x, y, sample_weights, max_depth, random_state):
        generator = self._wrap_generator(self._get_generator(x, y))
        tree_builder = self._get_tree_builder(
            x,
            y,
            sample_weights,
            generator,
            random_state,
            max_depth,
        )
        tree_builder.build_tree()
        self.tree_ = tree_builder.tree_


class BaseFeatureTreeRegressor(FeatureTreeMixin, BaseTreeRegressor):
    _parameter_constraints: dict = {
        **BaseTree._parameter_constraints,
        "criterion": [
            StrOptions(REG_CRITERION.keys()),
        ],
    }

    def _get_tree_builder(
        self, x, y, sample_weights, generator, random_state, max_depth
    ):
        Criterion = REG_CRITERION[self.criterion]
        return TreeBuilder(
            x,
            sample_weights,
            generator,
            Criterion(y),
            Tree(generator, 1),
            random_state,
            max_depth=max_depth,
            min_sample_split=self.min_samples_split,
            min_sample_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )


class BaseFeatureTreeClassifier(FeatureTreeMixin, BaseTreeClassifier):
    _parameter_constraints: dict = {
        **BaseTree._parameter_constraints,
        "criterion": [
            StrOptions(CLF_CRITERION.keys()),
        ],
        "class_weight": [
            StrOptions({"balanced"}),
            dict,
            None,
        ],
    }

    def _get_tree_builder(
        self, x, y, sample_weights, generator, random_state, max_depth
    ):
        Criterion = CLF_CRITERION[self.criterion]
        return TreeBuilder(
            x,
            sample_weights,
            generator,
            Criterion(y, self.n_classes_),
            Tree(generator, self.n_classes_),
            random_state,
            max_depth=max_depth,
            min_sample_split=self.min_samples_split,
            min_sample_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )


class DynamicTreeMixin:
    _parameter_constraints: dict = {"alpha": [float, None]}

    def _wrap_generator(self, generator):
        if hasattr(self, "alpha") and self.alpha is not None:
            if self.alpha == 0.0:
                raise ValueError("alpha == 0.0, must be != 0")

            return DynamicTreeAttributeGenerator(generator, self.alpha)

        return TreeAttributeGenerator(generator)


class ShapeletTreeRegressor(DynamicTreeMixin, ShapeletMixin, BaseFeatureTreeRegressor):
    """
    A shapelet tree regressor.

    Parameters
    ----------
    max_depth : int, optional
        The maximum depth of the tree. If ``None`` the tree is
        expanded until all leaves are pure or until all leaves contain less
        than ``min_samples_split`` samples.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        A split will be introduced only if the impurity decrease is larger
        than or equal to this value.
    n_shapelets : int, optional
        The number of shapelets to sample at each node.
    min_shapelet_size : float, optional
        The minimum length of a shapelets expressed as a fraction of
        *n_timestep*.
    max_shapelet_size : float, optional
        The maximum length of a shapelets expressed as a fraction of
        *n_timestep*.
    alpha : float, optional
        Dynamically decrease the number of sampled shapelets at each node according
        to the current depth, i.e.:

        ::
            w = 1 - exp(-abs(alpha) * depth)

        - if ``alpha < 0``, the number of sampled shapelets decrease from
            ``n_shapelets`` towards 1 with increased depth.
        - if ``alpha > 0``, the number of sampled shapelets increase from ``1``
            towards ``n_shapelets`` with increased depth.
        - if ``None``, the number of sampled shapelets are the same
            independeth of depth.
    metric : str or list, optional
        - If ``str``, the distance metric used to identify the best
            shapelet.
        - If ``list``, multiple metrics specified as a list of
            tuples, where the first element of the tuple is a metric name and
            the second element a dictionary with a parameter grid
            specification. A parameter grid specification is a dict with two
            mandatory and one optional key-value pairs defining the lower and
            upper bound on the values and number of values in the grid. For
            example, to specifiy a grid over the argument ``r`` with 10
            values in the range 0 to 1, we would give the following
            specification: ``dict(min_r=0, max_r=1, num_r=10)``.

            Read more about metric specifications in the `User guide
            <metric_specification>`__

        .. versionchanged:: 1.2
            Added support for multi-metric shapelet transform
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the `User guide
        <list_of_subsequence_metrics>`__.
    criterion : {"squared_error"}, optional
        The criterion used to evaluate the utility of a split.

        .. deprecated:: 1.1
            Criterion "mse" was deprecated in v1.1 and removed in version 1.2.
    random_state : int or RandomState
        - If ``int``, ``random_state`` is the seed used by the
            random number generator
        - If :class:`numpy.random.RandomState` instance, ``random_state``
            is the random number generator
        - If ``None``, the random number generator is the
            :class:`numpy.random.RandomState` instance used by
            :func:`numpy.random`.

    Attributes
    ----------
    tree_ : Tree
        The internal tree representation
    """

    _parameter_constraints: dict = {
        **BaseFeatureTreeRegressor._parameter_constraints,
        **ShapeletMixin._parameter_constraints,
        **DynamicTreeMixin._parameter_constraints,
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        n_shapelets="log2",
        min_shapelet_size=0,
        max_shapelet_size=1,
        alpha=None,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
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
        self.criterion = criterion
        self.alpha = alpha


class ExtraShapeletTreeRegressor(ShapeletTreeRegressor):
    """
    An extra shapelet tree regressor.

    Extra shapelet trees are constructed by sampling a distance threshold
    uniformly in the range [min(dist), max(dist)].

    Parameters
    ----------
    n_shapelets : int, optional
        The number of shapelets to sample at each node.
    max_depth : int, optional
        The maximum depth of the tree. If `None` the tree is expanded until all
        leaves are pure or until all leaves contain less than `min_samples_split`
        samples.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    criterion : {"squared_error"}, optional
        The criterion used to evaluate the utility of a split.

        .. deprecated:: 1.1
            Criterion "mse" was deprecated in v1.1 and removed in version 1.2.
    min_impurity_decrease : float, optional
        A split will be introduced only if the impurity decrease is larger than or
        equal to this value.
    min_shapelet_size : float, optional
        The minimum length of a sampled shapelet expressed as a fraction, computed
        as `min(ceil(X.shape[-1] * min_shapelet_size), 2)`.
    max_shapelet_size : float, optional
        The maximum length of a sampled shapelet, expressed as a fraction, computed
        as `ceil(X.shape[-1] * max_shapelet_size)`.
    metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
        Distance metric used to identify the best shapelet.
    metric_params : dict, optional
        Parameters for the distance measure.
    random_state : int or RandomState
        - If `int`, `random_state` is the seed used by the random number generator;
        - If `RandomState` instance, `random_state` is the random number generator;
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

    Attributes
    ----------
    tree_ : Tree
        The internal tree representation

    """

    _parameter_constraints: dict = {**ShapeletTreeRegressor._parameter_constraints}
    _parameter_constraints.pop("alpha")

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
        self, x, y, sample_weights, generator, random_state, max_depth
    ):
        Criterion = check_option(REG_CRITERION, self.criterion, "criterion")
        return ExtraTreeBuilder(
            x,
            sample_weights,
            generator,
            Criterion(y),
            Tree(generator, 1),
            random_state,
            max_depth=max_depth,
            min_sample_split=self.min_samples_split,
            min_sample_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )


class ShapeletTreeClassifier(
    DynamicTreeMixin, ShapeletMixin, BaseFeatureTreeClassifier
):
    """
    A shapelet tree classifier.

    Parameters
    ----------
    n_shapelets : int, optional
        The number of shapelets to sample at each node.
    max_depth : int, optional
        The maximum depth of the tree. If `None` the tree is expanded until all
        leaves are pure or until all leaves contain less than `min_samples_split`
        samples.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    criterion : {"entropy", "gini"}, optional
        The criterion used to evaluate the utility of a split.
    min_impurity_decrease : float, optional
        A split will be introduced only if the impurity decrease is larger than or
        equal to this value.
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
        - if None, each class has equal weight.
    random_state : int or RandomState
        - If `int`, `random_state` is the seed used by the random number generator;
        - If `RandomState` instance, `random_state` is the random number generator;
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

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

    _parameter_constraints: dict = {
        **BaseFeatureTreeClassifier._parameter_constraints,
        **ShapeletMixin._parameter_constraints,
        **DynamicTreeMixin._parameter_constraints,
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        n_shapelets="log2",
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
        self.criterion = criterion
        self.class_weight = class_weight
        self.alpha = alpha


class ExtraShapeletTreeClassifier(ShapeletTreeClassifier):
    """
    An extra shapelet tree classifier.

    Extra shapelet trees are constructed by sampling a distance threshold
    uniformly in the range ``[min(dist), max(dist)]``.

    Parameters
    ----------
    n_shapelets : int, optional
        The number of shapelets to sample at each node.
    max_depth : int, optional
        The maximum depth of the tree. If `None` the tree is expanded until all
        leaves are pure or until all leaves contain less than `min_samples_split`
        samples.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        A split will be introduced only if the impurity decrease is larger than or
        equal to this value.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_shapelet_size : float, optional
        The minimum length of a sampled shapelet expressed as a fraction, computed
        as `min(ceil(X.shape[-1] * min_shapelet_size), 2)`.
    max_shapelet_size : float, optional
        The maximum length of a sampled shapelet, expressed as a fraction, computed
        as `ceil(X.shape[-1] * max_shapelet_size)`.
    metric : {"euclidean", "scaled_euclidean", "dtw", "scaled_dtw"}, optional
        Distance metric used to identify the best shapelet.
    metric_params : dict, optional
        Parameters for the distance measure.
    criterion : {"entropy", "gini"}, optional
        The criterion used to evaluate the utility of a split.
    class_weight : dict or "balanced", optional
        Weights associated with the labels

        - if dict, weights on the form {label: weight}
        - if "balanced" each class weight inversely proportional to the class
            frequency
        - if None, each class has equal weight.
    random_state : int or RandomState, optional
        - If `int`, `random_state` is the seed used by the random number generator;
        - If `RandomState` instance, `random_state` is the random number generator;
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

    Attributes
    ----------
    tree_ : Tree
        The tree representation
    """

    _parameter_constraints: dict = {**ShapeletTreeClassifier._parameter_constraints}
    _parameter_constraints.pop("alpha")

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
        self, x, y, sample_weights, generator, random_state, max_depth
    ):
        Criterion = check_option(CLF_CRITERION, self.criterion, "criterion")
        return ExtraTreeBuilder(
            x,
            sample_weights,
            generator,
            Criterion(y, self.n_classes_),
            Tree(generator, self.n_classes_),
            random_state,
            max_depth=max_depth,
            min_sample_split=self.min_samples_split,
            min_sample_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )


class RocketTreeRegressor(RocketMixin, BaseFeatureTreeRegressor):
    """
    A tree regressor that uses random convolutions as features.

    Attributes
    ----------
    tree_ : Tree
        The internal tree representation.
    """

    _parameter_constraints: dict = {
        **BaseFeatureTreeRegressor._parameter_constraints,
        **RocketMixin._parameter_constraints,
        "random_state": ["random_state"],
    }

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
        """Construct a new rocket tree regressor.

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
        kernel_size : array-like, optional
            The kernel size, by default ``[7, 11, 13]``.
        min_size : float, optional
            The minimum timestep fraction to generate kernel sizes. If set,
            ``kernel_size`` cannot be set.
        max_size : float, optional
            The maximum timestep fractio to generate kernel sizes, If set,
            ``kernel_size`` cannot be set.
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


class RocketTreeClassifier(RocketMixin, BaseFeatureTreeClassifier):
    """A tree classifier that uses random convolutions as features.

    Attributes
    ----------
    tree_ : Tree
        The internal tree representation.

    """

    _parameter_constraints: dict = {
        **BaseFeatureTreeClassifier._parameter_constraints,
        **RocketMixin._parameter_constraints,
        "random_state": ["random_state"],
    }

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
        min_size=None,
        max_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        class_weight=None,
        random_state=None,
    ):
        """Construct a new rocket tree classifier.

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
        kernel_size : array-like, optional
            The kernel size, by default ``[7, 11, 13]``.
        min_size : float, optional
            The minimum timestep fraction to generate kernel sizes. If set,
            ``kernel_size`` cannot be set.
        max_size : float, optional
            The maximum timestep fractio to generate kernel sizes, If set,
            ``kernel_size`` cannot be set.
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
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.n_kernels = n_kernels
        self.criterion = criterion
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.min_size = min_size
        self.max_size = max_size
        self.bias_prob = bias_prob
        self.normalize_prob = normalize_prob
        self.padding_prob = padding_prob
        self.random_state = random_state
        self.class_weight = class_weight


class IntervalTreeClassifier(IntervalMixin, BaseFeatureTreeClassifier):
    """An interval based tree classifier.

    Attributes
    ----------
    tree_ : Tree
        The internal tree structure.

    """

    _parameter_constraints: dict = {
        **BaseFeatureTreeClassifier._parameter_constraints,
        **IntervalMixin._parameter_constraints,
        "random_state": ["random_state"],
    }

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
        """Construct a new interval tree classifier.

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
        self.class_weight = class_weight
        self.criterion = criterion


class IntervalTreeRegressor(IntervalMixin, BaseFeatureTreeRegressor):
    """An interval based tree regressor.

    Attributes
    ----------
    tree_ : Tree
        The internal tree structure.

    """

    _parameter_constraints: dict = {
        **BaseFeatureTreeRegressor._parameter_constraints,
        **IntervalMixin._parameter_constraints,
        "random_state": ["random_state"],
    }

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
        """Construct a new interval tree regressor.

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

            .. deprecated:: 1.1
                Criterion "mse" was deprecated in v1.1 and removed in version 1.2.
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
        self.criterion = criterion


class PivotTreeClassifier(PivotMixin, BaseFeatureTreeClassifier):
    """A tree classifier that uses pivot time series.

    Attributes
    ----------
    tree_ : Tree
        The internal tree representation

    """

    _parameter_constraints: dict = {
        **BaseFeatureTreeClassifier._parameter_constraints,
        **PivotMixin._parameter_constraints,
        "random_state": ["random_state"],
    }

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
        """Construct a new pivot tree classifier.

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
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.n_pivot = n_pivot
        self.metrics = metrics
        self.random_state = random_state
        self.criterion = criterion
        self.class_weight = class_weight


class DilatedShapeletTreeClassifier(DilatedShapeletMixin, BaseFeatureTreeClassifier):
    _parameter_constraints: dict = {
        **BaseFeatureTreeClassifier._parameter_constraints,
        **DilatedShapeletMixin._parameter_constraints,
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_shapelets=1000,
        *,
        metric="euclidean",
        metric_params=None,
        shapelet_size=None,
        min_shapelet_size=None,
        max_shapelet_size=None,
        normalize_prob=0.8,
        lower=0.05,
        upper=0.1,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="entropy",
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.n_shapelets = n_shapelets
        self.metric = metric
        self.metric_params = metric_params
        self.shapelet_size = shapelet_size
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.normalize_prob = normalize_prob
        self.lower = lower
        self.upper = upper
        self.criterion = criterion
        self.class_weight = class_weight
        self.random_state = random_state
