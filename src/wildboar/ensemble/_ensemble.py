# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import OutlierMixin
from sklearn.ensemble import BaggingClassifier as SklearnBaggingClassifier
from sklearn.ensemble import BaggingRegressor as SklearnBaggingRegressor
from sklearn.ensemble._bagging import BaseBagging as SklearnBaseBagging
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_scalar

from ..base import BaseEstimator
from ..transform._shapelet import RandomShapeletMixin
from ..tree import (
    ExtraShapeletTreeClassifier,
    ExtraShapeletTreeRegressor,
    IntervalTreeClassifier,
    IntervalTreeRegressor,
    PivotTreeClassifier,
    ProximityTreeClassifier,
    ShapeletTreeClassifier,
    ShapeletTreeRegressor,
)
from ..tree._tree import RocketTreeClassifier, RocketTreeRegressor


class BaseBagging(BaseEstimator, SklearnBaseBagging, metaclass=ABCMeta):
    """Base estimator for Wildboar ensemble estimators."""

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "max_samples": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0, 1, closed="right"),
        ],
        "bootstrap": ["boolean"],
        "oob_score": ["boolean"],
        "warm_start": ["boolean"],
        "n_jobs": [None, Integral],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
        "base_estimator": [
            HasMethods(["fit", "predict"]),
            StrOptions({"deprecated"}),
            None,
        ],
    }

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        bootstrap=True,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        base_estimator="deprecated",
    ):
        if base_estimator != "deprecated":
            warnings.warn(
                "base_estimator is deprecated and will be removed in 1.4",
                FutureWarning,
            )
            if estimator is None:
                estimator = base_estimator

        self.base_estimator = base_estimator
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=bootstrap,
            bootstrap_features=False,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _validate_data(
        self,
        x="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        **check_params,
    ):
        new_check_params = {"allow_3d": True}
        if "y_numeric" in check_params:
            new_check_params["y_numeric"] = check_params["y_numeric"]

        out = super()._validate_data(
            x,
            y,
            reset=reset,
            validate_separately=validate_separately,
            dtype=float,
            **new_check_params,
        )
        no_val_X = isinstance(x, str) and x == "no_validation"
        no_val_y = y is None or isinstance(y, str) and y == "no_validation"
        if no_val_X:
            return out
        elif no_val_y:
            x = out
        else:
            x, y = out

        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)

        if no_val_y:
            return x
        else:
            return x, y

    def fit(self, x, y, sample_weight=None):
        self._validate_params()
        x, y = self._validate_data(x, y, allow_3d=True, dtype=float)
        self._fit(
            x, y, self.max_samples, self.max_depth, sample_weight, check_input=False
        )
        return self

    def _make_estimator(self, append=True, random_state=None):
        estimator = super()._make_estimator(append, random_state)
        if self.n_dims_in_ > 1:
            estimator._force_n_dims = self.n_dims_in_
        return estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = True
        tags.input_tags.three_d_array = (
            self.estimator.__sklearn_tags__().input_tags.three_d_array
            if self.estimator is not None
            else True  # Default estimator support three_d_array
        )
        return tags


class ForestMixin:
    """
    Mixin for tree based ensembles.
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "bootstrap": ["boolean"],
        "oob_score": ["boolean"],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "warm_start": ["boolean"],
    }

    def apply(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, allow_3d=True, reset=False)
        results = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **{"prefer": "threads"},
        )(delayed(tree.apply)(x, check_input=False) for tree in self.estimators_)

        return np.array(results).T

    def decision_path(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, allow_3d=True, reset=False)
        indicators = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **{"prefer": "threads"},
        )(
            delayed(tree.decision_path)(x, check_input=False)
            for tree in self.estimators_
        )

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse.hstack(indicators).tocsr(), n_nodes_ptr


class BaggingClassifier(BaseBagging, SklearnBaggingClassifier):
    """
    A bagging classifier.

    A bagging regressor is a meta-estimator that fits base classifiers
    on random subsets of the original data.

    Parameters
    ----------
    estimator : object, optional
        Base estimator of the ensemble. If `None`, the base estimator
        is a :class:`~wildboar.tree.ShapeletTreeRegressor`.
    n_estimators : int, optional
        The number of base estimators in the ensemble.
    max_samples : int or float, optional
        The number of samples to draw from `X` to train each base estimator.

        - if `int`, then draw `max_samples` samples.
        - if `float`, then draw `max_samples * n_samples` samples.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    class_weight : dict or "balanced", optional
        Weights associated with the labels

        - if `dict`, weights on the form `{label: weight}`.
        - if "balanced" each class weight inversely proportional to
          the class frequency.
        - if `None`, each class has equal weight.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call
        to fit and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means
        using a single core and a value of `-1` means using all cores.
        Positive integers mean the exact number of cores.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the
          random number generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    verbose : int, optional
        Controls the output to standard error while fitting and predicting.
    base_estimator : object, optional
        Use `estimator` instead.

        .. deprecated:: 1.2
            `base_estimator` has been deprecated and will be removed in 1.4.
            Use `estimator` instead.
    """

    _parameter_constraints: dict = {**BaseBagging._parameter_constraints}

    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        bootstrap=True,
        oob_score=False,
        class_weight=None,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        base_estimator="deprecated",
    ):
        super().__init__(
            estimator=ShapeletTreeClassifier(strategy="random")
            if estimator is None
            else estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            base_estimator=base_estimator,
        )
        self.class_weight = class_weight

    def fit(self, x, y, sample_weight=None):
        self._validate_params()
        x, y = self._validate_data(x, y, allow_3d=True, dtype=float)

        if self.class_weight is not None:
            class_weight = compute_sample_weight(self.class_weight, y)
            if sample_weight is not None:
                sample_weight = sample_weight * class_weight
            else:
                sample_weight = class_weight

        self._fit(
            x,
            y,
            max_samples=self.max_samples,
            max_depth=self.max_depth,
            sample_weight=sample_weight,
            check_input=True,
        )
        return self

    def _get_estimator(self):
        if self.estimator is None:
            return ShapeletTreeClassifier(strategy="random")
        return self.estimator

    def predict_proba(self, X):
        X = self._validate_data(X, reset=False, allow_3d=True, dtype=float)
        return super().predict_proba(X)


class BaseForestClassifier(ForestMixin, BaggingClassifier, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators=100,
        estimator_params=tuple(),
        *,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        impurity_equality_tolerance=None,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.impurity_equality_tolerance = impurity_equality_tolerance
        self.class_weight = class_weight
        self.estimator_params = estimator_params

    def _fit(
        self,
        X,
        y,
        max_samples=None,
        max_depth=None,
        sample_weight=None,
        check_input=False,
    ):
        return super()._fit(
            X,
            y,
            max_samples=max_samples,
            max_depth=max_depth,
            sample_weight=sample_weight,
            check_input=False,
        )

    def _parallel_args(self):
        return {"prefer": "threads"}


class BaseShapeletForestClassifier(BaseForestClassifier, metaclass=ABCMeta):
    """Base class for shapelet forest classifiers.

    Warnings
    --------
    This class should not be used directly. Use derived classes instead.
    """

    @abstractmethod
    def __init__(  # noqa: PLR0913
        self,
        estimator,
        n_estimators=100,
        estimator_params=tuple(),
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        impurity_equality_tolerance=None,
        n_shapelets="log2",
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        coverage_probability=None,
        variability=1,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            estimator_params=estimator_params,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            impurity_equality_tolerance=impurity_equality_tolerance,
            criterion=criterion,
            class_weight=class_weight,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
        )
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.coverage_probability = coverage_probability
        self.variability = variability
        self.metric = metric
        self.metric_params = metric_params


class ShapeletForestClassifier(BaseShapeletForestClassifier):
    """
    An ensemble of random shapelet tree classifiers.

    A forest of randomized shapelet trees.

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators.
    n_shapelets : int, optional
        The number of shapelets to sample at each node.
    max_depth : int, optional
        The maximum depth of the tree. If `None` the tree is
        expanded until all leaves are pure or until all leaves contain less
        than `min_samples_split` samples.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        A split will be introduced only if the impurity decrease is larger
        than or equal to this value.
    impurity_equality_tolerance : float, optional
        Tolerance for considering two impurities as equal. If the impurity decrease
        is the same, we consider the split that maximizes the gap between the sum
        of distances.

        - If None, we never consider the separation gap.

        .. versionadded:: 1.3
    min_shapelet_size : float, optional
        The minimum length of a shapelets expressed as a fraction of
        *n_timestep*.
    max_shapelet_size : float, optional
        The maximum length of a shapelets expressed as a fraction of
        *n_timestep*.
    coverage_probability : float, optional
        The probability that a time step is covered by a
        shapelet, in the range 0 < coverage_probability <= 1.

        - For larger `coverage_probability`, we get larger shapelets.
        - For smaller `coverage_probability`, we get shorter shapelets.
    variability : float, optional
        Controls the shape of the Beta distribution used to
        sample shapelets. Defaults to 1.

        - Higher `variability` creates more uniform intervals.
        - Lower `variability` creates more variable intervals sizes.
    alpha : float, optional
        Dynamically decrease the number of sampled shapelets at each node
        according to the current depth, i.e.

            w = 1 - exp(-abs(alpha) * depth)

        - if `alpha < 0`, the number of sampled shapelets decrease from
          `n_shapelets` towards 1 with increased depth.
        - if `alpha > 0`, the number of sampled shapelets increase from `1`
          towards `n_shapelets` with increased depth.
        - if `None`, the number of sampled shapelets are the same
          independeth of depth.
    metric : str or list, optional
        The distance metric.

        - If `str`, the distance metric used to identify the best
          shapelet.
        - If `list`, multiple metrics specified as a list of tuples,
          where the first element of the tuple is a metric name and the second
          element a dictionary with a parameter grid specification. A parameter
          grid specification is a dict with two mandatory and one optional
          key-value pairs defining the lower and upper bound on the values and
          number of values in the grid. For example, to specifiy a grid over
          the argument `r` with 10 values in the range 0 to 1, we would give
          the following specification: `dict(min_r=0, max_r=1, num_r=10)`.

          Read more about metric specifications in the `User guide
          <metric_specification>`__.

        .. versionchanged:: 1.2
            Added support for multi-metric shapelet transform
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the `User guide
        <list_of_subsequence_metrics>`__.
    criterion : {"entropy", "gini"}, optional
        The criterion used to evaluate the utility of a split.
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call to
        fit and add more estimators to the ensemble, otherwise, just fit a
        whole new ensemble.
    class_weight : dict or "balanced", optional
        Weights associated with the labels

        - if `dict`, weights on the form `{label: weight}`.
        - if "balanced" each class weight inversely proportional to
          the class frequency.
        - if `None`, each class has equal weight.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means
        using a single core and a value of `-1` means using all cores.
        Positive integers mean the exact number of cores.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the
          random number generator
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.

    Examples
    --------
    >>> from wildboar.ensemble import ShapeletForestClassifier
    >>> from wildboar.datasets import load_synthetic_control
    >>> x, y = load_synthetic_control()
    >>> f = ShapeletForestClassifier(n_estimators=100, metric='scaled_euclidean')
    >>> f.fit(x, y)
    ShapeletForestClassifier(metric='scaled_euclidean')
    >>> y_hat = f.predict(x)
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **ShapeletTreeClassifier._parameter_constraints,
    }
    _parameter_constraints.pop("shapelet_size")
    _parameter_constraints.pop("sample_size")

    def __init__(  # noqa: PLR0913 # noqa: PLR0913
        self,
        n_estimators=100,
        *,
        n_shapelets="log2",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        impurity_equality_tolerance=None,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        alpha=None,
        coverage_probability=None,
        variability=1,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        class_weight=None,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            estimator=ShapeletTreeClassifier(strategy="random"),
            estimator_params=(
                "max_depth",
                "n_shapelets",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "impurity_equality_tolerance",
                "min_shapelet_size",
                "max_shapelet_size",
                "coverage_probability",
                "variability",
                "alpha",
                "metric",
                "metric_params",
                "criterion",
            ),
            n_shapelets=n_shapelets,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            impurity_equality_tolerance=impurity_equality_tolerance,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            coverage_probability=coverage_probability,
            variability=variability,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            oob_score=oob_score,
            bootstrap=bootstrap,
            warm_start=warm_start,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.alpha = alpha


class ExtraShapeletTreesClassifier(BaseShapeletForestClassifier):
    """
    An ensemble of extremely random shapelet trees.

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators.
    max_depth : int, optional
        The maximum depth of the tree. If `None` the tree is expanded
        until all leaves are pure or until all leaves contain less than
        `min_samples_split` samples.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        A split will be introduced only if the impurity decrease is larger than
        or equal to this value.
    min_shapelet_size : float, optional
        The minimum length of a shapelets expressed as a fraction of
        *n_timestep*.
    max_shapelet_size : float, optional
        The maximum length of a shapelets expressed as a fraction of
        *n_timestep*.
    coverage_probability : float, optional
        The probability that a time step is covered by a
        shapelet, in the range 0 < coverage_probability <= 1.

        - For larger `coverage_probability`, we get larger shapelets.
        - For smaller `coverage_probability`, we get shorter shapelets.
    variability : float, optional
        Controls the shape of the Beta distribution used to
        sample shapelets. Defaults to 1.

        - Higher `variability` creates more uniform intervals.
        - Lower `variability` creates more variable intervals sizes.
    metric : str or list, optional
        The distance metric.

        - If `str`, the distance metric used to identify the best
          shapelet.
        - If `list`, multiple metrics specified as a list of tuples,
          where the first element of the tuple is a metric name and the second
          element a dictionary with a parameter grid specification. A parameter
          grid specification is a dict with two mandatory and one optional
          key-value pairs defining the lower and upper bound on the values and
          number of values in the grid. For example, to specifiy a grid over
          the argument `r` with 10 values in the range 0 to 1, we would give
          the following specification: `dict(min_r=0, max_r=1, num_r=10)`.

          Read more about metric specifications in the `User guide
          <metric_specification>`__.

        .. versionchanged:: 1.2
            Added support for multi-metric shapelet transform
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the `User guide
        <list_of_subsequence_metrics>`__.
    criterion : {"entropy", "gini"}, optional
        The criterion used to evaluate the utility of a split.
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call to
        fit and add more estimators to the ensemble, otherwise, just fit a
        whole new ensemble.
    class_weight : dict or "balanced", optional
        Weights associated with the labels

        - if `dict`, weights on the form `{label: weight}`.
        - if "balanced" each class weight inversely proportional to
          the class frequency.
        - if :class:`None`, each class has equal weight.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means
        using a single core and a value of `-1` means using all cores.
        Positive integers mean the exact number of cores.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the
          random number generator
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.

    Examples
    --------
    >>> from wildboar.ensemble import ExtraShapeletTreesClassifier
    >>> from wildboar.datasets import load_synthetic_control
    >>> x, y = load_synthetic_control()
    >>> f = ExtraShapeletTreesClassifier(n_estimators=100, metric='scaled_euclidean')
    >>> f.fit(x, y)
    ExtraShapeletTreesClassifier(metric='scaled_euclidean')
    >>> y_hat = f.predict(x)
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **ExtraShapeletTreeClassifier._parameter_constraints,
    }
    _parameter_constraints.pop("n_shapelets")

    def __init__(  # noqa: PLR0913 # noqa: PLR0913
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        coverage_probability=None,
        variability=1,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        class_weight=None,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            estimator=ExtraShapeletTreeClassifier(),
            estimator_params=(
                "max_depth",
                "min_impurity_decrease",
                "min_samples_leaf",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "coverage_probability",
                "variability",
                "metric",
                "metric_params",
                "criterion",
            ),
            n_shapelets=1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            coverage_probability=coverage_probability,
            variability=variability,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            oob_score=oob_score,
            bootstrap=bootstrap,
            warm_start=warm_start,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )


class BaggingRegressor(BaseBagging, SklearnBaggingRegressor):
    """
    A bagging regressor.

    A bagging regressor is a meta-estimator that fits base classifiers
    on random subsets of the original data.

    Parameters
    ----------
    estimator : object, optional
        Base estimator of the ensemble. If `None`, the base estimator
        is a :class:`~wildboar.tree.ShapeletTreeRegressor`.
    n_estimators : int, optional
        The number of base estimators in the ensemble.
    max_samples : int or float, optional
        The number of samples to draw from `X` to train each base estimator.

        - if `int`, then draw `max_samples` samples.
        - if `float`, then draw `max_samples * n_samples` samples.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call
        to fit and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means
        using a single core and a value of `-1` means using all cores.
        Positive integers mean the exact number of cores.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the random
          number generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    verbose : int, optional
        Controls the output to standard error while fitting and predicting.
    base_estimator : object, optional
        Use `estimator` instead.

        .. deprecated:: 1.2
            `base_estimator` has been deprecated and will be removed in 1.4.
            Use `estimator` instead.
    """

    _parameter_constraints: dict = {**BaseBagging._parameter_constraints}

    def __init__(
        self,
        estimator=None,
        n_estimators=100,
        *,
        max_samples=1.0,
        bootstrap=True,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        base_estimator="deprecated",
    ):
        super().__init__(
            estimator=ShapeletTreeRegressor(strategy="random")
            if estimator is None
            else estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            base_estimator=base_estimator,
        )

    def fit(self, x, y, sample_weight=None):
        x, y = self._validate_data(x, y, allow_3d=True, dtype=float, y_numeric=True)

        super()._fit(
            x,
            y,
            max_samples=self.max_samples,
            max_depth=self.max_depth,
            sample_weight=sample_weight,
            check_input=False,
        )
        return self

    def predict(self, X):
        X = self._validate_data(X, allow_3d=True, dtype=float, reset=False)
        return super().predict(X)

    def _get_estimator(self):
        if self.estimator is None:
            return ShapeletTreeRegressor(strategy="random")
        return self.estimator


class BaseForestRegressor(ForestMixin, BaggingRegressor, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        *,
        estimator,
        estimator_params=tuple(),
        oob_score=False,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        impurity_equality_tolerance=None,
        criterion="squared_error",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
        )
        self.estimator_params = estimator_params
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.impurity_equality_tolerance = impurity_equality_tolerance

    def _fit(
        self,
        X,
        y,
        max_samples=None,
        max_depth=None,
        sample_weight=None,
        check_input=False,
    ):
        return super()._fit(X, y, max_samples, max_depth, sample_weight, False)

    def _parallel_args(self):
        return {"prefer": "threads"}


class BaseShapeletForestRegressor(BaseForestRegressor):
    """
    Base class for shapelet forest classifiers.

    Warnings
    --------
    This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        estimator,
        estimator_params=tuple(),
        oob_score=False,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        impurity_equality_tolerance=None,
        n_shapelets="log2",
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        coverage_probability=None,
        variability=1,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            estimator_params=estimator_params,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            impurity_equality_tolerance=impurity_equality_tolerance,
            criterion=criterion,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
        )
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.coverage_probability = coverage_probability
        self.variability = variability
        self.metric = metric
        self.metric_params = metric_params

    def _parallel_args(self):
        return {"prefer": "threads"}


class ShapeletForestRegressor(BaseShapeletForestRegressor):
    """
    An ensemble of random shapelet tree regressors.

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators.
    n_shapelets : int, optional
        The number of shapelets to sample at each node.
    max_depth : int, optional
        The maximum depth of the tree. If `None` the tree is
        expanded until all leaves are pure or until all leaves contain less
        than `min_samples_split` samples.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        A split will be introduced only if the impurity decrease is larger
        than or equal to this value.
    impurity_equality_tolerance : float, optional
        Tolerance for considering two impurities as equal. If the impurity decrease
        is the same, we consider the split that maximizes the gap between the sum
        of distances.

        - If None, we never consider the separation gap.

        .. versionadded:: 1.3
    min_shapelet_size : float, optional
        The minimum length of a shapelets expressed as a fraction of
        *n_timestep*.
    max_shapelet_size : float, optional
        The maximum length of a shapelets expressed as a fraction of
        *n_timestep*.
    coverage_probability : float, optional
        The probability that a time step is covered by a
        shapelet, in the range 0 < coverage_probability <= 1.

        - For larger `coverage_probability`, we get larger shapelets.
        - For smaller `coverage_probability`, we get shorter shapelets.
    variability : float, optional
        Controls the shape of the Beta distribution used to
        sample shapelets. Defaults to 1.

        - Higher `variability` creates more uniform intervals.
        - Lower `variability` creates more variable intervals sizes.
    alpha : float, optional
        Dynamically decrease the number of sampled shapelets at each node
        according to the current depth, i.e.

            w = 1 - exp(-abs(alpha) * depth)

        - if `alpha < 0`, the number of sampled shapelets decrease from
          `n_shapelets` towards 1 with increased depth.
        - if `alpha > 0`, the number of sampled shapelets increase from `1`
          towards `n_shapelets` with increased depth.
        - if `None`, the number of sampled shapelets are the same
          independeth of depth.
    metric : str or list, optional
        The distance metric.

        - If `str`, the distance metric used to identify the best
          shapelet.
        - If `list`, multiple metrics specified as a list of tuples,
          where the first element of the tuple is a metric name and the second
          element a dictionary with a parameter grid specification. A parameter
          grid specification is a dict with two mandatory and one optional
          key-value pairs defining the lower and upper bound on the values and
          number of values in the grid. For example, to specifiy a grid over
          the argument `r` with 10 values in the range 0 to 1, we would give
          the following specification: `dict(min_r=0, max_r=1, num_r=10)`.

          Read more about metric specifications in the `User guide
          <metric_specification>`__.

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
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call to
        fit and add more estimators to the ensemble, otherwise, just fit a
        whole new ensemble.
    n_jobs : int, optional
        The number of processor cores used for fitting the ensemble.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the
          random number generator
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.

    Examples
    --------
    >>> from wildboar.ensemble import ShapeletForestRegressor
    >>> from wildboar.datasets import load_synthetic_control
    >>> x, y = load_synthetic_control()
    >>> f = ShapeletForestRegressor(n_estimators=100, metric='scaled_euclidean')
    >>> f.fit(x, y)
    ShapeletForestRegressor(metric='scaled_euclidean')
    >>> y_hat = f.predict(x)
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **ShapeletTreeRegressor._parameter_constraints,
    }

    def __init__(  # noqa: PLR0913
        self,
        n_estimators=100,
        *,
        n_shapelets="log2",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        impurity_equality_tolerance=None,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        coverage_probability=None,
        variability=1,
        alpha=None,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            estimator=ShapeletTreeRegressor(strategy="random"),
            estimator_params=(
                "max_depth",
                "n_shapelets",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "impurity_equality_tolerance",
                "min_shapelet_size",
                "max_shapelet_size",
                "coverage_probability",
                "variability",
                "alpha",
                "metric",
                "metric_params",
                "criterion",
            ),
            n_shapelets=n_shapelets,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            impurity_equality_tolerance=impurity_equality_tolerance,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            coverage_probability=coverage_probability,
            variability=variability,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            oob_score=oob_score,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.alpha = alpha


class ExtraShapeletTreesRegressor(BaseShapeletForestRegressor):
    """
    An ensemble of extremely random shapelet tree regressors.

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators.
    max_depth : int, optional
        The maximum depth of the tree. If `None` the tree is expanded
        until all leaves are pure or until all leaves contain less than
        `min_samples_split` samples.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_shapelet_size : float, optional
        The minimum length of a shapelets expressed as a fraction of
        *n_timestep*.
    max_shapelet_size : float, optional
        The maximum length of a shapelets expressed as a fraction of
        *n_timestep*.
    coverage_probability : float, optional
        The probability that a time step is covered by a
        shapelet, in the range 0 < coverage_probability <= 1.

        - For larger `coverage_probability`, we get larger shapelets.
        - For smaller `coverage_probability`, we get shorter shapelets.
    variability : float, optional
        Controls the shape of the Beta distribution used to
        sample shapelets. Defaults to 1.

        - Higher `variability` creates more uniform intervals.
        - Lower `variability` creates more variable intervals sizes.
    metric : str or list, optional
        The distance metric.

        - If `str`, the distance metric used to identify the best
          shapelet.
        - If `list`, multiple metrics specified as a list of tuples,
          where the first element of the tuple is a metric name and the second
          element a dictionary with a parameter grid specification. A parameter
          grid specification is a dict with two mandatory and one optional
          key-value pairs defining the lower and upper bound on the values and
          number of values in the grid. For example, to specifiy a grid over
          the argument `r` with 10 values in the range 0 to 1, we would give
          the following specification: `dict(min_r=0, max_r=1, num_r=10)`.

          Read more about metric specifications in the `User guide
          <metric_specification>`__.

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
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call to
        fit and add more estimators to the ensemble, otherwise, just fit a
        whole new ensemble.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means
        using a single core and a value of `-1` means using all cores.
        Positive integers mean the exact number of cores.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If int, `random_state` is the seed used by the
          random number generator
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.

    Examples
    --------
    >>> from wildboar.ensemble import ExtraShapeletTreesRegressor
    >>> from wildboar.datasets import load_synthetic_control
    >>> x, y = load_synthetic_control()
    >>> f = ExtraShapeletTreesRegressor(n_estimators=100, metric='scaled_euclidean')
    >>> f.fit(x, y)
    ExtraShapeletTreesRegressor(metric='scaled_euclidean')
    >>> y_hat = f.predict(x)
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **ExtraShapeletTreeRegressor._parameter_constraints,
    }
    _parameter_constraints.pop("n_shapelets")

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_shapelet_size=0,
        max_shapelet_size=1,
        coverage_probability=None,
        variability=1,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            estimator=ExtraShapeletTreeRegressor(),
            estimator_params=(
                "max_depth",
                "min_impurity_decrease",
                "min_samples_leaf",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "coverage_probability",
                "variability",
                "metric",
                "metric_params",
                "criterion",
            ),
            n_shapelets=1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            coverage_probability=coverage_probability,
            variability=variability,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            oob_score=oob_score,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )


class ShapeletForestEmbedding(BaseShapeletForestRegressor):
    """
    An ensemble of random shapelet trees.

    An unsupervised transformation of a time series dataset to a
    high-dimensional sparse representation. A time series i indexed by the leaf
    that it falls into. This leads to a binary coding of a time series with as
    many ones as trees in the forest.

    The dimensionality of the resulting representation is `<= n_estimators *
    2^max_depth`

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators.
    n_shapelets : int, optional
        The number of shapelets to sample at each node.
    max_depth : int, optional
        The maximum depth of the tree. If `None` the tree is expanded
        until all leaves are pure or until all leaves contain less than
        `min_samples_split` samples.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        A split will be introduced only if the impurity decrease is larger than
        or equal to this value.
    min_shapelet_size : float, optional
        The minimum length of a shapelets expressed as a fraction of
        *n_timestep*.
    max_shapelet_size : float, optional
        The maximum length of a shapelets expressed as a fraction of
        *n_timestep*.
    coverage_probability : float, optional
        The probability that a time step is covered by a
        shapelet, in the range 0 < coverage_probability <= 1.

        - For larger `coverage_probability`, we get larger shapelets.
        - For smaller `coverage_probability`, we get shorter shapelets.
    variability : float, optional
        Controls the shape of the Beta distribution used to
        sample shapelets. Defaults to 1.

        - Higher `variability` creates more uniform intervals.
        - Lower `variability` creates more variable intervals sizes.
    metric : str or list, optional
        The distance metric.

        - If `str`, the distance metric used to identify the best
          shapelet.
        - If `list`, multiple metrics specified as a list of tuples,
          where the first element of the tuple is a metric name and the second
          element a dictionary with a parameter grid specification. A parameter
          grid specification is a dict with two mandatory and one optional
          key-value pairs defining the lower and upper bound on the values and
          number of values in the grid. For example, to specifiy a grid over
          the argument `r` with 10 values in the range 0 to 1, we would give
          the following specification: `dict(min_r=0, max_r=1, num_r=10)`.

          Read more about metric specifications in the `User guide
          <metric_specification>`__.

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
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call to
        fit and add more estimators to the ensemble, otherwise, just fit a
        whole new ensemble.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means
        using a single core and a value of `-1` means using all cores.
        Positive integers mean the exact number of cores.
    sparse_output : bool, optional
        Return a sparse CSR-matrix.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the
          random number generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **ExtraShapeletTreeRegressor._parameter_constraints,
        "sparse_output": ["boolean"],
    }

    def __init__(  # noqa: PLR0913
        self,
        n_estimators=100,
        *,
        n_shapelets=1,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        coverage_probability=None,
        variability=1,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        sparse_output=True,
        random_state=None,
    ):
        super().__init__(
            estimator=ExtraShapeletTreeRegressor(),
            estimator_params=(
                "n_shapelets",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "min_shapelet_size",
                "max_shapelet_size",
                "coverage_probability",
                "variability",
                "metric",
                "metric_params",
                "criterion",
            ),
            n_shapelets=n_shapelets,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            oob_score=False,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.sparse_output = sparse_output

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported")

    def fit(self, x, y=None, sample_weight=None):
        self.fit_transform(x, y, sample_weight)
        return self

    def fit_transform(self, x, y=None, sample_weight=None):
        x = self._validate_data(x, allow_3d=True, dtype=float)
        random_state = check_random_state(self.random_state)
        y = random_state.uniform(size=x.shape[0])
        super().fit(x, y, sample_weight=sample_weight)
        self.one_hot_encoder_ = OneHotEncoder(
            sparse=self.sparse_output, handle_unknown="ignore"
        )
        return self.one_hot_encoder_.fit_transform(self.apply(x))

    def transform(self, x):
        check_is_fitted(self)
        return self.one_hot_encoder_.transform(self.apply(x))


class IsolationShapeletForest(OutlierMixin, ForestMixin, BaseBagging):
    """
    An isolation shapelet forest.

    .. versionadded:: 0.3.5

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators in the ensemble.
    n_shapelets : int, optional
        The number of shapelets to sample at each node.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means using a
        single core and a value of `-1` means using all cores. Positive
        integers mean the exact number of cores.
    min_shapelet_size : float, optional
        The minimum length of a shapelets expressed as a fraction of
        *n_timestep*.
    max_shapelet_size : float, optional
        The maximum length of a shapelets expressed as a fraction of
        *n_timestep*.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    max_samples : "auto", float or int, optional
        The number of samples to draw to train each base estimator.
    contamination : 'auto' or float, optional
        The strategy for computing the offset.

        - if "auto" then `offset_` is set to `-0.5`.
        - if float `offset_` is computed as the c:th percentile of
          scores.

        If `bootstrap=True`, out-of-bag samples are used for computing
        the scores.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call to
        fit and add more estimators to the ensemble, otherwise, just fit a
        whole new ensemble.
    metric : str or list, optional
        The distance metric.

        - If `str`, the distance metric used to identify the best
          shapelet.
        - If `list`, multiple metrics specified as a list of tuples,
          where the first element of the tuple is a metric name and the second
          element a dictionary with a parameter grid specification. A parameter
          grid specification is a dict with two mandatory and one optional
          key-value pairs defining the lower and upper bound on the values and
          number of values in the grid. For example, to specifiy a grid over
          the argument `r` with 10 values in the range 0 to 1, we would give
          the following specification: `dict(min_r=0, max_r=1, num_r=10)`.

          Read more about metric specifications in the `User guide
          <metric_specification>`__.

        .. versionchanged:: 1.2
            Added support for multi-metric shapelet transform
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the `User guide
        <list_of_subsequence_metrics>`__.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the
          random number generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.

    Attributes
    ----------
    offset_ : float
        The offset for computing the final decision

    Examples
    --------
    Using default offset threshold

    >>> from wildboar.ensemble import IsolationShapeletForest
    >>> from wildboar.datasets import load_two_lead_ecg
    >>> from wildboar.model_selection import outlier_train_test_split
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> f = IsolationShapeletForest(random_state=1)
    >>> x, y = load_two_lead_ecg()
    >>> x_train, x_test, y_train, y_test = outlier_train_test_split(
    ...     x, y, 1, test_size=0.2, anomalies_train_size=0.05, random_state=1
    ... )
    >>> f.fit(x_train)
    IsolationShapeletForest(random_state=1)
    >>> y_pred = f.predict(x_test)
    >>> balanced_accuracy_score(y_test, y_pred) # doctest: +NUMBER
    0.8674
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **RandomShapeletMixin._parameter_constraints,
        "contamination": [StrOptions({"auto"}), Interval(Real, 0, 0.5, closed="right")],
        "max_samples": [
            None,
            Interval(Real, 0.0, 1.0, closed="right"),
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"auto"}),
        ],
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        n_shapelets=1,
        bootstrap=False,
        n_jobs=None,
        min_shapelet_size=0,
        max_shapelet_size=1,
        min_samples_split=2,
        max_samples="auto",
        contamination="auto",
        warm_start=False,
        metric="euclidean",
        metric_params=None,
        random_state=None,
    ):
        super(IsolationShapeletForest, self).__init__(
            estimator=ExtraShapeletTreeRegressor(),
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.estimator_params = (
            "n_shapelets",
            "min_samples_split",
            "min_shapelet_size",
            "max_shapelet_size",
            "metric",
            "metric_params",
        )
        self.n_shapelets = n_shapelets
        self.metric = metric
        self.metric_params = metric_params
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.min_samples_split = min_samples_split
        self.contamination = contamination
        self.max_samples = max_samples

    def _get_estimator(self):
        return self.estimator

    def _set_oob_score(self, x, y):
        raise NotImplementedError("OOB score not supported")

    def _parallel_args(self):
        return {"prefer": "threads"}

    def fit(self, x, y=None, sample_weight=None):
        x = self._validate_data(x, allow_3d=True, dtype=float)
        random_state = check_random_state(self.random_state)

        rnd_y = random_state.uniform(size=x.shape[0])
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(x.shape[0], 256)
            else:
                raise ValueError(
                    "max_samples must be 'auto', int or float, got %r."
                    % self.max_samples
                )
        elif isinstance(self.max_samples, numbers.Integral):
            max_samples = min(self.max_samples, x.shape[0])
        else:
            max_samples = math.ceil(
                check_scalar(
                    self.max_samples,
                    "max_samples",
                    numbers.Real,
                    min_val=0,
                    max_val=1.0,
                    include_boundaries="right",
                )
                * x.shape[0]
            )

        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super(IsolationShapeletForest, self)._fit(
            x,
            rnd_y,
            max_samples=max_samples,
            max_depth=max_depth,
            sample_weight=sample_weight,
            check_input=False,
        )

        self.max_samples_ = max_samples
        if self.contamination == "auto":
            self.offset_ = -0.5
        elif isinstance(self.contamination, numbers.Real):
            if not 0 < self.contamination <= 0.5:
                raise ValueError("contamination must be in (0, 0.5]")
            if self.bootstrap:
                scores = self._oob_score_samples(x)
            else:
                scores = self.score_samples(x)

            self.offset_ = np.percentile(scores, 100.0 * self.contamination)
        else:
            raise ValueError(
                "contamination must be 'auto' or float, got %r." % self.contamination
            )

        return self

    def predict(self, x):
        decision = self.decision_function(x)
        is_inlier = np.ones(decision.shape[0], dtype=int)
        is_inlier[decision < 0] = -1
        return is_inlier

    def decision_function(self, x):
        return self.score_samples(x) - self.offset_

    def score_samples(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, reset=False, allow_3d=True, dtype=float)
        return _score_samples(x, self.estimators_, self.max_samples_)

    def _oob_score_samples(self, x):
        n_samples = x.shape[0]
        score_samples = np.zeros((n_samples,))

        # We need to make this more efficient
        for i in range(x.shape[0]):
            estimators = []
            for estimator, samples in zip(self.estimators_, self.estimators_samples_):
                if i not in samples:
                    estimators.append(estimator)
            score_samples[i] = _score_samples(x[[i]], estimators, self.max_samples_)
        return score_samples

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.three_d_array = True
        return tags


def _score_samples(x, estimators, max_samples):
    # From: https://github.com/scikit-learn/scikit-learn/blob/
    # 0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/ensemble/_iforest.py#L411
    depths = np.zeros(x.shape[0], order="F")

    for tree in estimators:
        leaves_index = tree.apply(x, check_input=False)
        node_indicator = tree.decision_path(x, check_input=False)
        n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

        depths += (
            np.ravel(node_indicator.sum(axis=1))
            + _average_path_length(n_samples_leaf)
            - 1.0
        )

    denominator = len(estimators) * _average_path_length(np.array([max_samples]))
    scores = 2 ** -np.divide(
        depths, denominator, out=np.ones_like(depths), where=denominator != 0
    )
    return -scores


def _average_path_length(n_samples_leaf):
    # From: https://github.com/scikit-learn/scikit-learn/blob/
    # 0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/ensemble/_iforest.py#L480
    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)


class RocketForestRegressor(BaseForestRegressor):
    """
    An ensemble of rocket tree regressors.

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators.
    n_kernels : int, optional
        The number of shapelets to sample at each node.
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    max_depth : int, optional
        The maximum depth of the tree. If `None` the tree is
        expanded until all leaves are pure or until all leaves contain less
        than `min_samples_split` samples.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        A split will be introduced only if the impurity decrease is larger
        than or equal to this value.
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
        The minimum length of a shapelets expressed as a fraction of
        *n_timestep*.
    max_size : float, optional
        The maximum length of a shapelets expressed as a fraction of
        *n_timestep*.
    bias_prob : float, optional
        The probability of using a bias term.
    normalize_prob : float, optional
        The probability of performing normalization.
    padding_prob : float, optional
        The probability of padding with zeros.
    criterion : {"squared_error"}, optional
        The criterion used to evaluate the utility of a split.

        .. deprecated:: 1.1
            Criterion "mse" was deprecated in v1.1 and removed in version 1.2.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call to
        fit and add more estimators to the ensemble, otherwise, just fit a
        whole new ensemble.
    n_jobs : int, optional
        The number of processor cores used for fitting the ensemble.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the
          random number generator
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **RocketTreeRegressor._parameter_constraints,
    }

    def __init__(  # noqa: PLR0913
        self,
        n_estimators=100,
        *,
        n_kernels=10,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        impurity_equality_tolerance=None,
        sampling="normal",
        sampling_params=None,
        kernel_size=None,
        min_size=None,
        max_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        criterion="squared_error",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            estimator=RocketTreeRegressor(),
            estimator_params=(
                "n_kernels",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "kernel_size",
                "min_size",
                "max_size",
                "sampling",
                "sampling_params",
                "bias_prob",
                "normalize_prob",
                "padding_prob",
                "criterion",
            ),
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
        )
        self.n_kernels = n_kernels
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.bias_prob = bias_prob
        self.normalize_prob = normalize_prob
        self.padding_prob = padding_prob
        self.min_size = min_size
        self.max_size = max_size

    def _parallel_args(self):
        return {"prefer": "threads"}


class RocketForestClassifier(BaseForestClassifier):
    """
    An ensemble of rocket tree classifiers.

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators.
    n_kernels : int, optional
        The number of shapelets to sample at each node.
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    max_depth : int, optional
        The maximum depth of the tree. If `None` the tree is
        expanded until all leaves are pure or until all leaves contain less
        than `min_samples_split` samples.
    min_samples_split : int, optional
        The minimum number of samples to split an internal node.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        A split will be introduced only if the impurity decrease is larger
        than or equal to this value.
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
        The minimum length of a shapelets expressed as a fraction of
        *n_timestep*.
    max_size : float, optional
        The maximum length of a shapelets expressed as a fraction of
        *n_timestep*.
    bias_prob : float, optional
        The probability of using a bias term.
    normalize_prob : float, optional
        The probability of performing normalization.
    padding_prob : float, optional
        The probability of padding with zeros.
    criterion : {"entropy", "gini"}, optional
        The criterion used to evaluate the utility of a split.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call to
        fit and add more estimators to the ensemble, otherwise, just fit a
        whole new ensemble.
    class_weight : dict or "balanced", optional
        Weights associated with the labels

        - if `dict`, weights on the form `{label: weight}`.
        - if "balanced" each class weight inversely proportional to
          the class frequency.
        - if :class:`None`, each class has equal weight.
    n_jobs : int, optional
        The number of processor cores used for fitting the ensemble.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the
          random number generator
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **RocketTreeClassifier._parameter_constraints,
    }

    def __init__(  # noqa: PLR0913
        self,
        n_estimators=100,
        *,
        n_kernels=10,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        sampling="normal",
        sampling_params=None,
        kernel_size=None,
        min_size=None,
        max_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        class_weight=None,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            estimator=RocketTreeClassifier(),
            estimator_params=(
                "n_kernels",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "kernel_size",
                "min_size",
                "max_size",
                "sampling",
                "sampling_params",
                "bias_prob",
                "normalize_prob",
                "padding_prob",
                "criterion",
            ),
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
            class_weight=class_weight,
        )
        self.n_kernels = n_kernels
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.bias_prob = bias_prob
        self.normalize_prob = normalize_prob
        self.padding_prob = padding_prob
        self.min_size = min_size
        self.max_size = max_size

    def _parallel_args(self):
        return {"prefer": "threads"}


class IntervalForestClassifier(BaseForestClassifier):
    """
    An ensemble of interval tree classifiers.

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators.
    n_intervals : str, int or float, optional
        The number of intervals to use for the transform.

        - if "log2", the number of intervals is `log2(n_timestep)`.
        - if "sqrt", the number of intervals is `sqrt(n_timestep)`.
        - if int, the number of intervals is `n_intervals`.
        - if float, the number of intervals is `n_intervals * n_timestep`, with
          `0 < n_intervals < 1`.

        .. deprecated:: 1.2
            The option "log" has been renamed to "log2".
    intervals : str, optional
        The method for selecting intervals.

        - if "fixed", `n_intervals` non-overlapping intervals.
        - if "random", `n_intervals` possibly overlapping intervals of randomly
          sampled in `[min_size * n_timestep, max_size * n_timestep]`.

        .. deprecated:: 1.3
            The option "sample" has been deprecated. Use "fixed" with
            `sample_size`.
    summarizer : str or list, optional
        The method to summarize each interval.

        - if str, the summarizer is determined by `_SUMMARIZERS.keys()`.
        - if list, the summarizer is a list of functions `f(x) -> float`, where
          `x` is a numpy array.

        The default summarizer summarizes each interval as its mean, standard
        deviation and slope.
    sample_size : float, optional
        The sub-sample fixed intervals.
    min_size : float, optional
        The minimum interval size if `intervals="random"`.
    max_size : float, optional
        The maximum interval size if `intervals="random"`.
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    max_depth : int, optional
        The maximum tree depth.
    min_samples_split : int, optional
        The minimum number of samples to consider a split.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        The minimum impurity decrease to build a sub-tree.
    criterion : {"entropy", "gini"}, optional
        The impurity criterion.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call
        to fit and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means
        using a single core and a value of `-1` means using all cores.
        Positive integers mean the exact number of cores.
    class_weight : dict or "balanced", optional
        Weights associated with the labels.

        - if dict, weights on the form {label: weight}.
        - if "balanced" each class weight inversely proportional to the class
            frequency.
        - if None, each class has equal weight.
    random_state : int or RandomState, optional
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
          by `np.random`.
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **IntervalTreeClassifier._parameter_constraints,
    }

    def __init__(  # noqa: PLR0913
        self,
        n_estimators=100,
        *,
        n_intervals="sqrt",
        intervals="random",
        summarizer="mean_var_slope",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            estimator=IntervalTreeClassifier(),
            estimator_params=(
                "n_intervals",
                "intervals",
                "summarizer",
                "sample_size",
                "min_size",
                "max_size",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "criterion",
            ),
            oob_score=oob_score,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.n_intervals = n_intervals
        self.intervals = intervals
        self.summarizer = summarizer
        self.sample_size = sample_size
        self.min_size = min_size
        self.max_size = max_size

    def _parallel_args(self):
        return {"prefer": "threads"}


class IntervalForestRegressor(BaseForestRegressor):
    """
    An ensemble of interval tree regressors.

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators.
    n_intervals : str, int or float, optional
        The number of intervals to use for the transform.

        - if "log2", the number of intervals is `log2(n_timestep)`.
        - if "sqrt", the number of intervals is `sqrt(n_timestep)`.
        - if int, the number of intervals is `n_intervals`.
        - if float, the number of intervals is `n_intervals * n_timestep`, with
          `0 < n_intervals < 1`.

        .. deprecated:: 1.2
            The option "log" has been renamed to "log2".
    intervals : str, optional
        The method for selecting intervals.

        - if "fixed", `n_intervals` non-overlapping intervals.
        - if "sample", `n_intervals * sample_size` non-overlapping intervals.
        - if "random", `n_intervals` possibly overlapping intervals of randomly
          sampled in `[min_size * n_timestep, max_size * n_timestep]`.
    summarizer : str or list, optional
        The method to summarize each interval.

        - if str, the summarizer is determined by `_SUMMARIZERS.keys()`.
        - if list, the summarizer is a list of functions `f(x) -> float`, where
          `x` is a numpy array.

        The default summarizer summarizes each interval as its mean, variance
        and slope.
    sample_size : float, optional
        The sample size of fixed intervals if `intervals="sample"`.
    min_size : float, optional
        The minimum interval size if `intervals="random"`.
    max_size : float, optional
        The maximum interval size if `intervals="random"`.
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    max_depth : int, optional
        The maximum tree depth.
    min_samples_split : int, optional
        The minimum number of samples to consider a split.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        The minimum impurity decrease to build a sub-tree.
    criterion : {"entropy", "gini"}, optional
        The impurity criterion.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call
        to fit and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means
        using a single core and a value of `-1` means using all cores.
        Positive integers mean the exact number of cores.
    random_state : int or RandomState, optional
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
          by `np.random`.
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **IntervalTreeRegressor._parameter_constraints,
    }

    def __init__(  # noqa: PLR0913
        self,
        n_estimators=100,
        *,
        n_intervals="sqrt",
        intervals="fixed",
        summarizer="mean_var_slope",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        criterion="squared_error",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            estimator=IntervalTreeRegressor(),
            estimator_params=(
                "n_intervals",
                "intervals",
                "summarizer",
                "sample_size",
                "min_size",
                "max_size",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "criterion",
            ),
            oob_score=oob_score,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.n_intervals = n_intervals
        self.intervals = intervals
        self.summarizer = summarizer
        self.sample_size = sample_size
        self.min_size = min_size
        self.max_size = max_size

    def _parallel_args(self):
        return {"prefer": "threads"}


class PivotForestClassifier(BaseForestClassifier):
    """An ensemble of interval tree classifiers."""

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **PivotTreeClassifier._parameter_constraints,
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        n_pivot="sqrt",
        metrics="all",
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            estimator=PivotTreeClassifier(),
            estimator_params=(
                "n_pivot",
                "metrics",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "criterion",
            ),
            oob_score=oob_score,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.n_pivot = n_pivot
        self.metrics = metrics

    def _parallel_args(self):
        return {"prefer": "threads"}


class ProximityForestClassifier(BaseForestClassifier):
    """
    A forest of proximity trees.

    Parameters
    ----------
    n_estimators : int, optional
        The number of estimators.
    n_pivot : int, optional
        The number of pivots to sample at each node.
    pivot_sample : {"label", "uniform"}, optional
        The pivot sampling method.
    metric_sample : {"uniform", "weighted"}, optional
        The metric sampling method.
    metric : {"auto", "default"}, str or list, optional
        The distance metrics. By default, we use the parameterization suggested by
        Lucas et.al (2019).

        - If "auto", use the default metric specification, suggested by
          (Lucas et. al, 2020).
        - If str, use a single metric or default metric specification.
        - If list, custom metric specification can be given as a list of
          tuples, where the first element of the tuple is a metric name and the
          second element a dictionary with a parameter grid specification. A
          parameter grid specification is a `dict` with two mandatory and one
          optional key-value pairs defining the lower and upper bound on the
          values as well as the number of values in the grid. For example, to
          specifiy a grid over the argument 'r' with 10 values in the range 0
          to 1, we would give the following specification:
          `dict(min_r=0, max_r=1, num_r=10)`.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_metrics>`.
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the :ref:`User guide
        <list_of_metrics>`.
    metric_factories : dict, optional
        A metric specification.

        .. deprecated:: 1.2
            Use the combination of metric and metric params.
    oob_score : bool, optional
        Use out-of-bag samples to estimate generalization performance. Requires
        `bootstrap=True`.
    max_depth : int, optional
        The maximum tree depth.
    min_samples_split : int, optional
        The minimum number of samples to consider a split.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        The minimum impurity decrease to build a sub-tree.
    criterion : {"entropy", "gini"}, optional
        The impurity criterion.
    bootstrap : bool, optional
        If the samples are drawn with replacement.
    warm_start : bool, optional
        When set to `True`, reuse the solution of the previous call
        to fit and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means
        using a single core and a value of `-1` means using all cores.
        Positive integers mean the exact number of cores.
    class_weight : dict or "balanced", optional
        Weights associated with the labels.

        - if dict, weights on the form {label: weight}.
        - if "balanced" each class weight inversely proportional to the class
            frequency.
        - if None, each class has equal weight.
    random_state : int or RandomState, optional
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

    References
    ----------
    Lucas, Benjamin, Ahmed Shifaz, Charlotte Pelletier, Lachlan O'Neill, Nayyar Zaidi, \
    Bart Goethals, François Petitjean, and Geoffrey I. Webb. (2019)
        Proximity forest: an effective and scalable distance-based classifier for time
        series. Data Mining and Knowledge Discovery
    """

    _parameter_constraints: dict = {
        **ForestMixin._parameter_constraints,
        **ProximityTreeClassifier._parameter_constraints,
    }

    def __init__(  # noqa: PLR0913
        self,
        n_estimators=100,
        *,
        n_pivot=1,
        pivot_sample="label",
        metric_sample="weighted",
        metric="auto",
        metric_params=None,
        metric_factories=None,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            estimator=ProximityTreeClassifier(),
            estimator_params=(
                "n_pivot",
                "metric_factories",
                "metric",
                "metric_params",
                "pivot_sample",
                "metric_sample",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "criterion",
            ),
            oob_score=oob_score,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.n_pivot = n_pivot
        self.metric_factories = metric_factories
        self.metric = metric
        self.metric_params = metric_params
        self.pivot_sample = pivot_sample
        self.metric_sample = metric_sample

    def _parallel_args(self):
        return {"prefer": "threads"}
