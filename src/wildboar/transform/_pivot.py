# Authors: Isak Samsten
# License: BSD 3 clause
import numbers

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions

from ..base import BaseEstimator
from ..distance._distance import _METRICS
from ..distance._multi_metric import make_metrics
from ..utils._rand import RandomSampler
from ..utils.validation import check_option
from ._base import BaseAttributeTransform
from ._cpivot import PivotAttributeGenerator

_METRIC_NAMES = set(_METRICS.keys())
_METRIC_NAMES.add("auto")
_METRIC_NAMES = frozenset(_METRIC_NAMES)


class PivotMixin:
    """Mixin for Pivot based estimators."""

    _parameter_constraints: dict = {
        "n_pivots": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric": [StrOptions(_METRIC_NAMES), list],
        "metric_params": [None, dict],
        "metric_sample": [StrOptions({"uniform", "weighted"}), None],
    }

    def _get_generator(self, x, y):
        if isinstance(self.metric, str) and self.metric == "auto":
            metric_specs = [
                ("euclidean", None),
                ("dtw", None),
                ("ddtw", None),
                ("dtw", dict(min_r=0.0, max_r=0.25, num_r=10)),
                ("ddtw", dict(min_r=0.0, max_r=0.25, num_r=10)),
                ("wdtw", dict(min_g=0.2, max_g=1.0)),
                ("wddtw", dict(min_g=0.2, max_g=1.0)),
                (
                    "lcss",
                    dict(
                        min_r=0.0,
                        max_r=0.25,
                        min_epsilon=0.2,
                        max_epsilon=1.0,
                    ),
                ),
                ("erp", dict(min_g=0, max_g=1.0)),
                ("msm", dict(min_c=0.01, max_c=100, num_c=50)),
                (
                    "twe",
                    dict(
                        min_penalty=0.00001,
                        max_penalty=1.0,
                        min_stiffness=0.000001,
                        max_stiffness=0.1,
                    ),
                ),
            ]
            metrics, weights = make_metrics(metric_specs)
        elif isinstance(self.metric, str):
            Metric = check_option(_METRICS, self.metric, "metric")
            metric_params = self.metric_params if self.metric_params is not None else {}
            metrics = [Metric(**metric_params)]
            weights = np.ones(1)
        else:
            metrics, weights = make_metrics(self.metric)

        if self.metric_sample is None or self.metric_sample == "uniform":
            weights = None

        return PivotAttributeGenerator(
            self.n_pivots, metrics, RandomSampler(len(metrics), weights)
        )


class PivotTransform(PivotMixin, BaseAttributeTransform):
    """
    A transform using pivot time series and sampled distance metrics.

    Parameters
    ----------
    n_pivots : int, optional
        The number of pivot time series.
    metric : {'auto'} or list, optional
        - If str, the metric to compute the distance.
        - If list, multiple metrics specified as a list of tuples, where the first
            element of the tuple is a metric name and the second element a dictionary
            with a parameter grid specification. A parameter grid specification is a
            dict with two mandatory and one optional key-value pairs defining the
            lower and upper bound on the values and number of values in the grid. For
            example, to specifiy a grid over the argument 'r' with 10 values in the
            range 0 to 1, we would give the following specification: `dict(min_r=0,
            max_r=1, num_r=10)`.

        Read more about the metrics and their parameters in the :ref:`User guide
        <list_of_subsequence_metrics>`.
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the :ref:`User guide <list_of_metrics>`.
    metric_sample : {"uniform", "weighted"}, optional
        If multiple metrics are specified this parameter controls how they are
        sampled. "uniform" samples each metric configuration with equal probability
        and "weighted" samples each metric with equal probability. By default,
        metric configurations are sampled with equal probability.
    random_state : int or np.RandomState, optional
        The random state.
    n_jobs : int, optional
        The number of cores to use.
    """

    _parameter_constraints: dict = {
        **PivotMixin._parameter_constraints,
        **BaseAttributeTransform._parameter_constraints,
    }

    def __init__(
        self,
        n_pivots=100,
        *,
        metric="auto",
        metric_params=None,
        metric_sample=None,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.n_pivots = n_pivots
        self.metric = metric
        self.metric_params = metric_params
        self.metric_sample = metric_sample


class ProximityTransform(TransformerMixin, BaseEstimator):
    """
    Transform time series based on class conditional pivots.

    Parameters
    ----------
    n_pivots : int, optional
        The number of pivot time series per class.
    metric : {'auto'} or list, optional
        - If str, the metric to compute the distance.
        - If list, multiple metrics specified as a list of tuples, where the first
            element of the tuple is a metric name and the second element a dictionary
            with a parameter grid specification. A parameter grid specification is a
            dict with two mandatory and one optional key-value pairs defining the
            lower and upper bound on the values and number of values in the grid. For
            example, to specifiy a grid over the argument 'r' with 10 values in the
            range 0 to 1, we would give the following specification: `dict(min_r=0,
            max_r=1, num_r=10)`.

        Read more about the metrics and their parameters in the :ref:`User guide
        <list_of_subsequence_metrics>`.
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the :ref:`User guide <list_of_metrics>`.
    metric_sample : {"uniform", "weighted"}, optional
        If multiple metrics are specified this parameter controls how they are
        sampled. "uniform" samples each metric configuration with equal probability
        and "weighted" samples each metric with equal probability. By default,
        metric configurations are sampled with equal probability.
    random_state : int or np.RandomState, optional
        The random state.
    n_jobs : int, optional
        The number of cores to use.
    """

    def __init__(
        self,
        n_pivots=100,
        metric="auto",
        metric_params=None,
        metric_sample="weighted",
        random_state=None,
        n_jobs=None,
    ):
        self.n_pivots = n_pivots
        self.metric = metric
        self.metric_params = metric_params
        self.metric_sample = metric_sample
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = self._validate_data(X, y, allow_3d=True, dtype=float)
        random_state = check_random_state(self.random_state)
        self.pivots_ = [
            PivotTransform(
                n_pivots=int(min(self.n_pivots, count)),
                metric=self.metric,
                metric_params=self.metric_params,
                metric_sample=self.metric_sample,
                random_state=random_state.randint(np.iinfo(np.int32).max),
                n_jobs=self.n_jobs,
            ).fit(X[y == label])
            for label, count in zip(*np.unique(y, return_counts=True))
        ]

        return self

    def transform(self, X, y=None):
        return np.hstack([pivots.transform(X) for pivots in self.pivots_])
