# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers
import warnings

import numpy as np
from sklearn.utils import resample
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
)
from sklearn.utils.deprecation import deprecated

from ..distance._distance import _METRICS, _SUBSEQUENCE_METRICS
from ..distance._matrix_profile import matrix_profile
from ..distance._multi_metric import make_subsequence_metrics
from ._base import BaseAttributeTransform
from ._cshapelet import (
    CastorAttributeGenerator,
    CastorSummarizer,
    CoverageProbabilityMultiMetricShapeletAttributeGenerator,
    CoverageProbabilityShapeletAttributeGenerator,
    DilatedShapeletAttributeGenerator,
    FixedShapeletAttributeGenerator,
    RandomMultiMetricShapeletAttributeGenerator,
    RandomShapeletAttributeGenerator,
)


def _select_shapelets(
    X,
    y,
    shapelet_size=11,
    n_shapelets=1,
    n_samples=1.0,
    random_state=None,
    n_jobs=None,
):
    if not hasattr(shapelet_size, "__iter__"):
        shapelet_size = [shapelet_size]

    labels, label_inverse = np.unique(y, return_inverse=True)

    shapelet_start = np.empty(
        (X.shape[0], len(shapelet_size) * n_shapelets), dtype=np.intp
    )
    for pad, window in enumerate(shapelet_size):
        selfjoin = [
            matrix_profile(X[y == label], window=window, kind="default", n_jobs=n_jobs)
            for label in labels
        ]

        for i in range(len(labels)):
            label = labels[i]
            X_label = X[label == y]
            not_label = label != y
            X_not_label = resample(
                X[not_label],
                replace=False,
                n_samples=max(1, math.ceil(not_label.sum() * n_samples)),
                random_state=random_state,
            )

            ab_join = matrix_profile(
                X_label, X_not_label, window=window, kind="default", n_jobs=n_jobs
            )
            shapelet_start[
                label_inverse == i,
                (pad * n_shapelets) : (pad * n_shapelets) + n_shapelets,
            ] = np.argpartition(ab_join - selfjoin[i], -n_shapelets)[:, -n_shapelets:]

    return shapelet_start


_COMMON_PARAMETER_CONSTRAINTS = {
    "n_shapelets": [
        Interval(numbers.Integral, 1, None, closed="left"),
        StrOptions({"log2", "sqrt", "auto"}),
        callable,
    ],
    "metric": [
        StrOptions(_SUBSEQUENCE_METRICS.keys()),
        list,
        dict,
    ],
    "metric_params": [dict, None],
}


class FixedShapeletMixin:
    _parameter_constraints = {
        **_COMMON_PARAMETER_CONSTRAINTS,
        "shapelet_size": [
            Interval(numbers.Integral, 1, None, closed="left"),
            Interval(numbers.Real, 0.0, 1.0, closed="right"),
            "array-like",
        ],
        "sample_size": [Interval(numbers.Real, 0.0, 1.0, closed="right")],
    }

    def _get_generator(self, X, y):  # noqa: PLR0912
        def _shapelet_size(shapelet_size):
            if not isinstance(shapelet_size, numbers.Integral):
                shapelet_size = math.ceil(X.shape[-1] * shapelet_size)
                shapelet_size = max(shapelet_size, 1)

            if not 0 <= shapelet_size < X.shape[-1]:
                raise ValueError(
                    "shapelet_size must be larger than 0 and less than n_timestep_in_"
                )

            return shapelet_size

        if isinstance(self.n_shapelets, str) and self.n_shapelets == "auto":
            n_shapelets = 1
        elif isinstance(self.n_shapelets, str) or callable(self.n_shapelets):
            if self.n_shapelets == "log2":
                n_shapelets = int(np.log2(X.shape[0]))
            elif self.n_shapelets == "sqrt":
                n_shapelets = int(np.sqrt(X.shape[0]))
            else:
                n_shapelets = int(self.n_shapelets(X.shape[0]))
        else:
            n_shapelets = self.n_shapelets

        if hasattr(self.shapelet_size, "__iter__"):
            shapelet_size = [_shapelet_size(s) for s in self.shapelet_size]
        else:
            shapelet_size = _shapelet_size(self.shapelet_size)

        self._shapelet_start = _select_shapelets(
            X,
            y,
            shapelet_size=shapelet_size,
            n_shapelets=n_shapelets,
            n_samples=self.sample_size,
            random_state=self.random_state if hasattr(self, "random_state") else None,
            n_jobs=self.n_jobs if hasattr(self, "n_jobs") else 1,
        )
        self._shapelet_length = np.broadcast_to(
            np.repeat(shapelet_size, n_shapelets).astype(np.intp),
            shape=self._shapelet_start.shape,
        )

        if isinstance(self.metric, str):
            return FixedShapeletAttributeGenerator(
                _SUBSEQUENCE_METRICS[self.metric](
                    **(self.metric_params if self.metric_params is not None else {})
                ),
                self._shapelet_start,
                self._shapelet_length,
            )
        else:
            raise ValueError(
                "When setting strategy='best', only a single metric is allowed"
            )


class RandomShapeletMixin:
    """Mixin for shapelet based estimators."""

    _parameter_constraints: dict = {
        **_COMMON_PARAMETER_CONSTRAINTS,
        "min_shapelet_size": [
            Interval(numbers.Real, 0, 1, closed="both"),
        ],
        "max_shapelet_size": [
            Interval(numbers.Real, 0, 1, closed="both"),
        ],
        "coverage_probability": [
            Interval(numbers.Real, 0, 1, closed="left"),
            None,
        ],
        "variability": [
            Interval(numbers.Real, 0, None, closed="left"),
            None,
        ],
    }

    def _get_generator(self, x, y):  # noqa: PLR0912
        if (
            not hasattr(self, "coverage_probability")
            or self.coverage_probability is None
        ):
            if self.min_shapelet_size > self.max_shapelet_size:
                raise ValueError(
                    f"The min_shapelet_size parameter of {type(self).__qualname__} "
                    "must be <= max_shapelet_size."
                )

            max_shapelet_size = math.ceil(self.n_timesteps_in_ * self.max_shapelet_size)
            min_shapelet_size = math.ceil(self.n_timesteps_in_ * self.min_shapelet_size)
            if min_shapelet_size < max_shapelet_size:
                possible_shapelets = sum(
                    self.n_timesteps_in_ - curr_len + 1
                    for curr_len in range(min_shapelet_size, max_shapelet_size)
                )
            else:
                possible_shapelets = self.n_timesteps_in_ - min_shapelet_size + 1
        else:
            possible_shapelets = self.n_timesteps_in_

        possible_shapelets *= self.n_dims_in_

        if isinstance(self.n_shapelets, str) and self.n_shapelets == "auto":
            n_shapelets = 1000
        elif isinstance(self.n_shapelets, str) or callable(self.n_shapelets):
            if self.n_shapelets == "log2":
                n_shapelets = int(np.log2(possible_shapelets))
            elif self.n_shapelets == "sqrt":
                n_shapelets = int(np.sqrt(possible_shapelets))
            else:
                n_shapelets = int(self.n_shapelets(possible_shapelets))
        else:
            n_shapelets = self.n_shapelets

        if (
            hasattr(self, "coverage_probability")
            and self.coverage_probability is not None
        ):
            if hasattr(self, "variability"):
                variability = self.variability if self.variability is not None else 1
            else:
                variability = 1

            if isinstance(self.metric, str):
                metric_params = (
                    self.metric_params if self.metric_params is not None else {}
                )
                return CoverageProbabilityShapeletAttributeGenerator(
                    _SUBSEQUENCE_METRICS[self.metric](**metric_params),
                    self.coverage_probability,
                    variability,
                    max(1, n_shapelets),
                )
            else:
                metrics, weights = make_subsequence_metrics(self.metric)
                return CoverageProbabilityMultiMetricShapeletAttributeGenerator(
                    max(1, n_shapelets),
                    self.coverage_probability,
                    self.variability,
                    metrics,
                    weights,
                )
        else:
            if min_shapelet_size < 2:
                # NOTE: To ensure that the same random_seed generates the same shapelets
                # in future versions we keep the limit of 2 timesteps for a shapelet as long
                # as the time series is at least 2 timesteps. Otherwise we fall back to 1
                # timestep.
                if self.n_timesteps_in_ < 2:
                    min_shapelet_size = 1
                else:
                    min_shapelet_size = 2

            if isinstance(self.metric, str):
                metric_params = (
                    self.metric_params if self.metric_params is not None else {}
                )
                return RandomShapeletAttributeGenerator(
                    _SUBSEQUENCE_METRICS[self.metric](**metric_params),
                    min_shapelet_size,
                    max_shapelet_size,
                    max(1, n_shapelets),
                )
            else:
                metrics, weights = make_subsequence_metrics(self.metric)
                return RandomMultiMetricShapeletAttributeGenerator(
                    max(1, n_shapelets),
                    min_shapelet_size,
                    max_shapelet_size,
                    metrics,
                    weights,
                )


class ShapeletMixin(RandomShapeletMixin, FixedShapeletMixin):
    _parameter_constraints = {
        "strategy": [StrOptions({"random", "best", "warn"})],
        **RandomShapeletMixin._parameter_constraints,
        **FixedShapeletMixin._parameter_constraints,
    }

    def _get_generator(self, x, y):  # noqa: PLR0912
        # If the attributes _shapelet_start or _shapelet_length
        # are set, we assume the user wants to use the FixedShapeletMixin.
        if hasattr(self, "strategy") and self.strategy == "best":
            return FixedShapeletMixin._get_generator(self, x, y)
        else:
            # TODO(1.4)
            if hasattr(self, "strategy") and self.strategy == "warn":
                warnings.warn(
                    "The value for 'strategy' will change to 'best' in 1.4. "
                    "Change 'strategy' to 'random' to keep the current behavior.",
                    FutureWarning,
                )

            # Silently default to RandomShapeletMixin if strategy is not set
            # or if it is set to "random" or "warn". Assuming the parameters
            # are validated.
            return RandomShapeletMixin._get_generator(self, x, y)


def _odd_shapelet_size(shapelet_size, n_timesteps):
    if shapelet_size == 1:
        shapelet_size = 3

    if shapelet_size % 2 == 0:
        shapelet_size += 1
    return shapelet_size


class DilatedShapeletMixin:
    _parameter_constraints = {
        "n_shapelets": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric_params": [dict, None],
        "metric": [StrOptions(_METRICS.keys())],
        "min_shapelet_size": [None, Interval(numbers.Real, 0, 1, closed="both")],
        "max_shapelet_size": [None, Interval(numbers.Real, 0, 1, closed="both")],
        "shapelet_size": [None, "array-like"],
        "normalize_prob": [Interval(numbers.Real, 0, 1, closed="both")],
        "lower": [Interval(numbers.Real, 0, 1, closed="both")],
        "upper": [Interval(numbers.Real, 0, 1, closed="both")],
        "ignore_y": ["boolean"],
    }

    def _get_generator(self, x, y):
        metric_params = self.metric_params if self.metric_params is not None else {}
        Metric = _METRICS[self.metric]

        if self.min_shapelet_size is not None or self.max_shapelet_size:
            if self.shapelet_size is not None:
                raise ValueError(
                    "Both shapelet_size and min_shapelet_size or max_shapelet_size "
                    "cannot be set at the same time"
                )

            min_shapelet_size = (
                self.min_shapelet_size if self.min_shapelet_size is not None else 0
            )
            max_shapelet_size = (
                self.max_shapelet_size if self.max_shapelet_size is not None else 1
            )
            if min_shapelet_size > max_shapelet_size:
                raise ValueError(
                    "max_shapelet_size must be larger than min_shapelet_size"
                )

            min_shapelet_size = int(self.n_timesteps_in_ * min_shapelet_size)
            max_shapelet_size = int(self.n_timesteps_in_ * max_shapelet_size)
            min_shapelet_size = max(min_shapelet_size, 2)
            max_shapelet_size = max(max_shapelet_size, 3)
            shapelet_size = range(min_shapelet_size, max_shapelet_size)
        elif self.shapelet_size is None:
            shapelet_size = [7, 9, 11]
        else:
            shapelet_size = self.shapelet_size

        shapelet_size = np.array(
            [_odd_shapelet_size(size, x.shape[-1]) for size in shapelet_size]
        )

        if self.lower > self.upper:
            raise ValueError("Lower can't be larger than upper")

        if y is not None and (not hasattr(self, "ignore_y") or not self.ignore_y):
            _, y, samples_per_label = np.unique(
                np.array(y), return_inverse=True, return_counts=True
            )
            samples = np.argsort(y, kind="stable")
        else:
            samples = None
            samples_per_label = None
            y = None

        return DilatedShapeletAttributeGenerator(
            Metric(**metric_params),
            self.n_shapelets,
            shapelet_size,
            self.normalize_prob,
            self.lower,
            self.upper,
            y,
            samples,
            samples_per_label,
        )


class CastorMixin:
    _parameter_constraints = {
        "n_groups": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_shapelets": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric_params": [dict, None],
        "metric": [StrOptions(_METRICS.keys())],
        "shapelet_size": [Interval(numbers.Integral, 3, None, closed="left")],
        "normalize_prob": [Interval(numbers.Real, 0, 1, closed="both")],
        "lower": [Interval(numbers.Real, 0, 1, closed="both")],
        "upper": [Interval(numbers.Real, 0, 1, closed="both")],
        "soft_min": [bool],
        "soft_max": [bool],
        "soft_threshold": [bool],
        "ignore_y": ["boolean"],
    }

    def _get_generator(self, x, y):
        Metric = _METRICS[self.metric]
        metric_params = self.metric_params if self.metric_params is not None else {}
        if y is not None and (not hasattr(self, "ignore_y") or not self.ignore_y):
            _, y, samples_per_label = np.unique(
                np.array(y), return_inverse=True, return_counts=True
            )
            samples = np.argsort(y, kind="stable")
        else:
            samples = None
            samples_per_label = None
            y = None

        return CastorAttributeGenerator(
            self.n_groups,
            self.n_shapelets,
            _odd_shapelet_size(self.shapelet_size, x.shape[-1]),
            self.normalize_prob,
            self.lower,
            self.upper,
            Metric(**metric_params),
            CastorSummarizer(self.soft_min, self.soft_max, self.soft_threshold),
            y,
            samples,
            samples_per_label,
        )


class CastorTransform(CastorMixin, BaseAttributeTransform):
    """
    Competing Dialated Shapelet Transform.

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
    ignore_y : bool, optional
        Ignore `y` and use the same sample which a shapelet is sampled from to
        estimate the distance threshold.
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


    Notes
    -----
    For better performance with multivariate datasets, set `n_shapelets` to
    `n_shapelets * n_dims` to ensure feature variability.
    """

    _parameter_constraints = {
        **CastorMixin._parameter_constraints,
        **BaseAttributeTransform._parameter_constraints,
    }

    def __init__(
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
        soft_min=True,
        soft_max=False,
        soft_threshold=True,
        ignore_y=False,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(n_jobs=n_jobs, random_state=random_state)
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
        self.ignore_y = ignore_y


class DilatedShapeletTransform(DilatedShapeletMixin, BaseAttributeTransform):
    """
    Dilated shapelet transform.

    Transform time series to a representation consisting of three
    values per shapelet: minimum dilated distance, the index of
    the timestep that minimizes the distance and number of subsequences
    that are below a distance threshold.

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
        The size of shapelets, by default [7, 9, 11].
    lower : float, optional
        The lower percentile to draw distance thresholds above.
    upper : float, optional
        The upper percentile to draw distance thresholds below.
    ignore_y : bool, optional
        Ignore `y` and use the same sample which a shapelet is sampled from to
        estimate the distance threshold.
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
        **DilatedShapeletMixin._parameter_constraints,
        **BaseAttributeTransform._parameter_constraints,
    }

    def __init__(
        self,
        n_shapelets=1000,
        *,
        metric="euclidean",
        metric_params=None,
        normalize_prob=0.5,
        min_shapelet_size=None,
        max_shapelet_size=None,
        shapelet_size=None,
        lower=0.05,
        upper=0.1,
        ignore_y=False,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.shapelet_size = shapelet_size
        self.metric = metric
        self.metric_params = metric_params
        self.normalize_prob = normalize_prob
        self.lower = lower
        self.upper = upper
        self.ignore_y = ignore_y


# TODO(1.4)
@deprecated(
    "RandomShapeletTransform will be removed in 1.4, use ShapeletTransform "
    "with `strategy='random' to retain the current behavior."
)
class RandomShapeletTransform(RandomShapeletMixin, BaseAttributeTransform):
    """
    Random shapelet tranform.

    Transform a time series to the distances to a selection of random
    shapelets.

    Parameters
    ----------
    n_shapelets : int, optional
        The number of shapelets in the resulting transform.
    metric : str or list, optional
        - If str, the distance metric used to identify the best shapelet.
        - If list, multiple metrics specified as a list of tuples, where the first
            element of the tuple is a metric name and the second element a dictionary
            with a parameter grid specification. A parameter grid specification is a
            dict with two mandatory and one optional key-value pairs defining the
            lower and upper bound on the values and number of values in the grid. For
            example, to specifiy a grid over the argument 'r' with 10 values in the
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
    n_jobs : int, optional
        The number of jobs to run in parallel. None means 1 and -1 means using all
        processors.
    random_state : int or RandomState, optional
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

    Attributes
    ----------
    embedding_ : Embedding
        The underlying embedding object.

    References
    ----------
    Wistuba, Martin, Josif Grabocka, and Lars Schmidt-Thieme.
        Ultra-fast shapelets for time series classification. arXiv preprint
        arXiv:1503.05018 (2015).

    Examples
    --------
    Transform each time series to the minimum DTW distance to each shapelet

    >>> from wildboar.dataset import load_gunpoint()
    >>> from wildboar.transform import RandomShapeletTransform
    >>> t = RandomShapeletTransform(metric="dtw")
    >>> t.fit_transform(X)

    Transform each time series to the either the minimum DTW distance, with r randomly
    set set between 0 and 1 or ERP distance with g between 0 and 1.

    >>> t = RandomShapeletTransform(
    ...     metric=[
    ...         ("dtw", dict(min_r=0.0, max_r=1.0)),
    ...         ("erp", dict(min_g=0.0, max_g=1.0)),
    ...     ]
    ... )
    >>> t.fit_transform(X)
    """

    _parameter_constraints: dict = {
        **RandomShapeletMixin._parameter_constraints,
        **BaseAttributeTransform._parameter_constraints,
    }

    def __init__(
        self,
        n_shapelets=1000,
        *,
        metric="euclidean",
        metric_params=None,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.n_shapelets = n_shapelets
        self.metric = metric
        self.metric_params = metric_params
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size


class ShapeletTransform(ShapeletMixin, BaseAttributeTransform):
    """
    Shapelet Transform.

    Transform a time series to the distances to a selection of shapelets. The
    transform is unsupervised if `strategy="random"` and supervised if
    `strategy="best"`.

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

        .. warning::
            Multiple metrics are only supported if `strategy="random"`.
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the :ref:`User guide
        <list_of_subsequence_metrics>`.
    strategy : {"best", "random"}, optional
        The strategy for selecting shapelets.

        - If "random", `n_shapelets` shapelets are randomly selected in the
          range defined by `min_shapelet_size` and `max_shapelet_size`
        - If "best", `n_shapelets` shapelets are selected per input sample
          of the size determined by `shapelet_size`.

        If strategy is set to "best", the transformation is supervised and requires
        `y`.
    shapelet_size : int, float or array-like, optional
        The shapelet size if `strategy="best"`.

        - If int, the exact shapelet size.
        - If float, a fraction of the number of input timestep.
        - If array-like, a list of float or int.
    sample_size : float, optional
        The size of the sample to determine the shapelets, if `shapelet_size="best"`.
    min_shapelet_size : float, optional
        Minimum shapelet size.
    max_shapelet_size : float, optional
        Maximum shapelet size.
    random_state : int or RandomState, optional
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.
    n_jobs : int, optional
        The number of jobs to run in parallel. None means 1 and -1 means using all
        processors.

    Attributes
    ----------
    embedding_ : Embedding
        The underlying embedding object.

    References
    ----------
    Wistuba, Martin, Josif Grabocka, and Lars Schmidt-Thieme.
        Ultra-fast shapelets for time series classification. arXiv preprint
        arXiv:1503.05018 (2015).

    Examples
    --------
    Transform each time series to the minimum DTW distance to each shapelet

    >>> from wildboar.dataset import load_gunpoint()
    >>> from wildboar.transform import ShapeletTransform
    >>> t = ShapeletTransform(metric="dtw")
    >>> t.fit_transform(X)

    Transform each time series to the either the minimum DTW distance, with r randomly
    set set between 0 and 1 or ERP distance with g between 0 and 1.

    >>> t = ShapeletTransform(
    ...     metric=[
    ...         ("dtw", dict(min_r=0.0, max_r=1.0)),
    ...         ("erp", dict(min_g=0.0, max_g=1.0)),
    ...     ]
    ... )
    >>> t.fit_transform(X)

    Transform each time series to the scaled euclidean distance between the most
    promising shapelet of size 38:

    >>> t = ShapeletTransform(strategy="best", shapelet_size=38)
    >>> t.fit_transform(X, y)
    """

    _parameter_constraints: dict = {
        **ShapeletMixin._parameter_constraints,
        **BaseAttributeTransform._parameter_constraints,
    }

    def __init__(
        self,
        n_shapelets="auto",
        *,
        metric="euclidean",
        metric_params=None,
        strategy="random",
        shapelet_size=0.1,
        sample_size=1.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        coverage_probability=None,
        variability=None,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.n_shapelets = n_shapelets
        self.metric = metric
        self.metric_params = metric_params
        self.strategy = strategy
        self.shapelet_size = shapelet_size
        self.sample_size = sample_size
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.coverage_probability = coverage_probability
        self.variability = variability

    def _more_tags(self):
        return {"requires_y": self.strategy == "best"}
