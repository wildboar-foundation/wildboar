# Authors: Isak Samsten
# License: BSD 3 clause
import numbers

import numpy as np
from joblib import effective_n_jobs
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval

from ..transform import (
    CastorTransform,
    DiffTransform,
    DilatedShapeletTransform,
    ShapeletTransform,
)
from ..transform._shapelet import RandomShapeletMixin
from ._transform import TransformRidgeClassifierCV, TransformRidgeCV


class RandomShapeletClassifier(TransformRidgeClassifierCV):
    """
    A classifier that uses random shapelets.

    Parameters
    ----------
    n_shapelets : int or {"log2", "sqrt", "auto"}, optional
        The number of shapelets in the resulting transform.

        - if, "auto" the number of shapelets depend on the value of `strategy`.
          For "best" the number is 1; and for "random" it is 1000.
        - if, "log2", the number of shaplets is the log2 of the total possible
          number of shapelets.
        - if, "sqrt", the number of shaplets is the square root of the total
          possible number of shapelets.
    metric : str or list, optional
        - If str, the distance metric used to identify the best shapelet.
        - If list, multiple metrics specified as a list of tuples, where the first
            element of the tuple is a metric name and the second element a dictionary
            with a parameter grid specification. A parameter grid specification is a
            dict with two mandatory and one optional key-value pairs defining the
            lower and upper bound on the values and number of values in the grid. For
            example, to specify a grid over the argument 'r' with 10 values in the
            range 0 to 1, we would give the following specification: ``dict(min_r=0,
            max_r=1, num_r=10)``.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the :ref:`User guide
        <list_of_subsequence_metrics>`.
    min_shapelet_size : float, optional
        Minimum shapelet size.
    max_shapelet_size : float, optional
        Maximum shapelet size.
    alphas : array-like of shape (n_alphas,), optional
        Array of alpha values to try.
    fit_intercept : bool, optional
        Whether to calculate the intercept for this model.
    normalize : bool, optional
        Standardize before fitting.
    scoring : str, callable, optional
        A string or a scorer callable object with signature
        `scorer(estimator, X, y)`.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form `{class_label: weight}`.
    random_state : int or RandomState, optional
        Controls the random sampling of kernels.

        - If `int`, `random_state` is the seed used by the random number
          generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    n_jobs : int, optional
        The number of parallel jobs.

    References
    ----------
    Wistuba, Martin, Josif Grabocka, and Lars Schmidt-Thieme.
        Ultra-fast shapelets for time series classification. arXiv preprint
        arXiv:1503.05018 (2015).
    """

    _parameter_constraints = {
        **TransformRidgeClassifierCV._parameter_constraints,
        **RandomShapeletMixin._parameter_constraints,
    }

    def __init__(
        self,
        n_shapelets=1000,
        *,
        metric="euclidean",
        metric_params=None,
        min_shapelet_size=0.1,
        max_shapelet_size=1.0,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        normalize=False,
        scoring=None,
        cv=None,
        class_weight=None,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            normalize=normalize,
            scoring=scoring,
            cv=cv,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.n_shapelets = n_shapelets
        self.metric = metric
        self.metric_params = metric_params
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size

    def _get_transform(self, random_state):
        return ShapeletTransform(
            self.n_shapelets,
            metric=self.metric,
            strategy="random",
            metric_params=self.metric_params,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            random_state=random_state,
            n_jobs=self.n_jobs,
        )


class RandomShapeletRegressor(TransformRidgeCV):
    """
    A regressor that uses random shapelets.

    Parameters
    ----------
    n_shapelets : int or {"log2", "sqrt", "auto"}, optional
        The number of shapelets in the resulting transform.

        - if, "auto" the number of shapelets depend on the value of `strategy`.
          For "best" the number is 1; and for "random" it is 1000.
        - if, "log2", the number of shaplets is the log2 of the total possible
          number of shapelets.
        - if, "sqrt", the number of shaplets is the square root of the total
          possible number of shapelets.
    metric : str or list, optional
        - If str, the distance metric used to identify the best shapelet.
        - If list, multiple metrics specified as a list of tuples, where the first
            element of the tuple is a metric name and the second element a dictionary
            with a parameter grid specification. A parameter grid specification is a
            dict with two mandatory and one optional key-value pairs defining the
            lower and upper bound on the values and number of values in the grid. For
            example, to specify a grid over the argument 'r' with 10 values in the
            range 0 to 1, we would give the following specification: ``dict(min_r=0,
            max_r=1, num_r=10)``.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the :ref:`User guide
        <list_of_subsequence_metrics>`.
    min_shapelet_size : float, optional
        Minimum shapelet size.
    max_shapelet_size : float, optional
        Maximum shapelet size.
    alphas : array-like of shape (n_alphas,), optional
        Array of alpha values to try.
    fit_intercept : bool, optional
        Whether to calculate the intercept for this model.
    normalize : bool, optional
        Standardize before fitting.
    scoring : str, callable, optional
        A string or a scorer callable object with signature
        `scorer(estimator, X, y)`.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    gcv_mode : {'auto', 'svd', 'eigen'}, optional
        Flag indicating which strategy to use when performing
        Leave-One-Out Cross-Validation. Options are::

            'auto' : use 'svd' if n_samples > n_features, otherwise use 'eigen'
            'svd' : force use of singular value decomposition of X when X is
                dense, eigenvalue decomposition of X^T.X when X is sparse.
            'eigen' : force computation via eigendecomposition of X.X^T

        The 'auto' mode is the default and is intended to pick the cheaper
        option of the two depending on the shape of the training data.
    random_state : int or RandomState, optional
        Controls the random sampling of kernels.

        - If `int`, `random_state` is the seed used by the random number
          generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    n_jobs : int, optional
        The number of parallel jobs.

    References
    ----------
    Wistuba, Martin, Josif Grabocka, and Lars Schmidt-Thieme.
        Ultra-fast shapelets for time series classification. arXiv preprint
        arXiv:1503.05018 (2015).
    """

    _parameter_constraints = {
        **TransformRidgeCV._parameter_constraints,
        **RandomShapeletMixin._parameter_constraints,
    }

    def __init__(
        self,
        n_shapelets=1000,
        *,
        metric="euclidean",
        metric_params=None,
        min_shapelet_size=0.1,
        max_shapelet_size=1.0,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        normalize=False,
        scoring=None,
        cv=None,
        gcv_mode=None,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            normalize=normalize,
            scoring=scoring,
            cv=cv,
            gcv_mode=gcv_mode,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.n_shapelets = n_shapelets
        self.metric = metric
        self.metric_params = metric_params
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size

    def _get_transform(self, random_state):
        return ShapeletTransform(
            self.n_shapelets,
            metric=self.metric,
            strategy="random",
            metric_params=self.metric_params,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            random_state=random_state,
            n_jobs=self.n_jobs,
        )


class DilatedShapeletClassifier(TransformRidgeClassifierCV):
    """
    A classifier that uses random dilated shapelets.

    Parameters
    ----------
    n_shapelets : int, optional
        The number of dilated shapelets.
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.
    normalize_prob : float, optional
        The probability of standardizing a shapelet with zero mean and unit
        standard deviation.
    min_shapelet_size : float, optional
        The minimum shapelet size. If None, use the discrete sizes
        in `shapelet_size`.
    max_shapelet_size : float, optional
        The maximum shapelet size. If None, use the discrete sizes
        in `shapelet_size`.
    shapelet_size : array-like, optional
        The size of shapelets.
    lower : float, optional
        The lower percentile to draw distance thresholds above.
    upper : float, optional
        The upper percentile to draw distance thresholds below.
    alphas : array-like of shape (n_alphas,), optional
        Array of alpha values to try.
    fit_intercept : bool, optional
        Whether to calculate the intercept for this model.
    scoring : str, callable, optional
        A string or a scorer callable object with signature
        `scorer(estimator, X, y)`.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form `{class_label: weight}`.
    normalize : bool, optional
        Standardize before fitting.
    random_state : int or RandomState, optional
        Controls the random sampling of kernels.

        - If `int`, `random_state` is the seed used by the random number
          generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    n_jobs : int, optional
        The number of parallel jobs.

    References
    ----------
    Antoine Guillaume, Christel Vrain, Elloumi Wael
        Random Dilated Shapelet Transform: A New Approach for Time Series Shapelets
        Pattern Recognition and Artificial Intelligence, 2022
    """

    _parameter_constraints = {
        **TransformRidgeClassifierCV._parameter_constraints,
        **DilatedShapeletTransform._parameter_constraints,
    }
    _parameter_constraints.pop("ignore_y")

    def __init__(  # noqa: PLR0913
        self,
        n_shapelets=1000,
        *,
        metric="euclidean",
        metric_params=None,
        normalize_prob=0.8,
        min_shapelet_size=None,
        max_shapelet_size=None,
        shapelet_size=None,
        lower=0.05,
        upper=0.1,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        scoring=None,
        cv=None,
        class_weight=None,
        normalize=True,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            normalize=normalize,
            scoring=scoring,
            cv=cv,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
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

    def _get_transform(self, random_state):
        return DilatedShapeletTransform(
            n_shapelets=self.n_shapelets,
            metric=self.metric,
            metric_params=self.metric_params,
            shapelet_size=self.shapelet_size,
            normalize_prob=self.normalize_prob,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            lower=self.lower,
            upper=self.upper,
            random_state=random_state,
            n_jobs=self.n_jobs,
        )


class CastorClassifier(TransformRidgeClassifierCV):
    """
    A dictionary based method using dilated competing shapelets.

    Parameters
    ----------
    n_groups : int, optional
        The number of groups of dilated shapelets.
    n_shapelets : int, optional
        The number of dilated shapelets per group.
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.
    normalize_prob : float, optional
        The probability of standardizing a shapelet with zero mean and unit
        standard deviation.
    shapelet_size : int, optional
        The length of the dilated shapelet.
    lower : float, optional
        The lower percentile to draw distance thresholds above.
    upper : float, optional
        The upper percentile to draw distance thresholds below.
    order : int or array-like, optional
        The order of difference.

        If int, half the groups with corresponding shapelets will be convolved
        with the `order` discrete difference along the time dimension.
    soft_min : bool, optional
        If `True`, use the sum of minimal distances. Otherwise, use the count
        of minimal distances.
    soft_max : bool, optional
        If `True`, use the sum of maximal distances. Otherwise, use the count
        of maximal distances.
    soft_threshold : bool, optional
        If `True`, count the time steps below the threshold for all shapelets.
        Otherwise, count the time steps below the threshold for the shapelet
        with the minimal distance.
    alphas : array-like of shape (n_alphas,), optional
        Array of alpha values to try.
    fit_intercept : bool, optional
        Whether to calculate the intercept for this model.
    scoring : str, callable, optional
        A string or a scorer callable object with signature
        `scorer(estimator, X, y)`.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form `{class_label: weight}`.
    normalize : "sparse" or bool, optional
        Standardize before fitting. By default use
        :class:`datasets.preprocess.SparseScaler` to standardize the attributes. Set
        to `False` to disable or `True` to use `StandardScaler`.
    random_state : int or RandomState, optional
        Controls the random sampling of kernels.

        - If `int`, `random_state` is the seed used by the random number
          generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is
          the random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    n_jobs : int, optional
        The number of parallel jobs.
    """

    _parameter_constraints = {
        **TransformRidgeClassifierCV._parameter_constraints,
        **CastorTransform._parameter_constraints,
        "order": [Interval(numbers.Integral, 0, None, closed="left")],
    }
    _parameter_constraints.pop("ignore_y")

    def __init__(  # noqa: PLR0913
        self,
        n_groups=64,
        n_shapelets=8,
        *,
        metric="euclidean",
        metric_params=None,
        normalize_prob=0.8,
        shapelet_size=11,
        lower=0.05,
        upper=0.1,
        order=1,
        soft_min=True,
        soft_max=False,
        soft_threshold=True,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        scoring=None,
        cv=None,
        class_weight=None,
        normalize="sparse",
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            normalize=normalize,
            scoring=scoring,
            cv=cv,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.n_groups = n_groups
        self.n_shapelets = n_shapelets
        self.metric = metric
        self.metric_params = metric_params
        self.normalize_prob = normalize_prob
        self.shapelet_size = shapelet_size
        self.lower = lower
        self.upper = upper
        self.soft_min = soft_min
        self.soft_max = soft_max
        self.soft_threshold = soft_threshold
        self.order = order

    def _get_transform(self, random_state):  # noqa: PLR0912
        random_state = check_random_state(random_state)
        params = dict(
            n_shapelets=self.n_shapelets,
            metric=self.metric,
            metric_params=self.metric_params,
            normalize_prob=self.normalize_prob,
            lower=self.lower,
            upper=self.upper,
            soft_min=self.soft_min,
            soft_max=self.soft_max,
            soft_threshold=self.soft_threshold,
            n_jobs=self.n_jobs,
        )
        n_jobs = effective_n_jobs(self.n_jobs)

        if (
            isinstance(self.order, numbers.Integral)
            and self.order > 0
            and self.n_groups > 1
        ):
            n_groups = self.n_groups // 2
            union_n_jobs = n_jobs // n_groups
            if union_n_jobs < 1:
                union_n_jobs = None

            return make_union(
                CastorTransform(
                    n_groups=n_groups,
                    shapelet_size=self.shapelet_size,
                    random_state=random_state.randint(np.iinfo(np.int32).max),
                    **params,
                ),
                make_pipeline(
                    DiffTransform(order=self.order),
                    CastorTransform(
                        n_groups=n_groups,
                        shapelet_size=self.shapelet_size,
                        random_state=random_state.randint(np.iinfo(np.int32).max),
                        **params,
                    ),
                ),
            )
        else:
            return CastorTransform(
                n_groups=self.n_groups,
                shapelet_size=self.shapelet_size,
                random_state=random_state,
                **params,
            )
