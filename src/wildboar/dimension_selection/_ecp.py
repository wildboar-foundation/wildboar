# Authors: Isak Samsten
# License: BSD 3 clause

import abc
import numbers

import numpy as np
from sklearn.base import _fit_context
from sklearn.validation._parameter_constraints import Interval, StrOptions

from ..base import BaseEstimator
from ..distance._distance import pairwise_distance
from ..distance.dtw import dtw_average
from ..utils.validation import MetricOptions, check_classification_targets
from ._base import DimensionSelectorMixin


class _Prototype(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def create(self, X):
        pass


class _DtwPrototype(_Prototype):
    def __init__(self, r):
        self.r = r

    def create(self, X):
        return dtw_average(X, r=self.r)


class _MeanPrototype(_Prototype):
    def create(self, X):
        return np.mean(X, axis=0)


class _MedianPrototype(_Prototype):
    def create(self, X):
        return np.median(X, axis=0)


_PROTOTYPE = {"mean": _MeanPrototype, "median": _MeanPrototype, "dtw": _DtwPrototype}


class ECSSelector(DimensionSelectorMixin, BaseEstimator):
    """
    ElboxClassSum (ECS) dimension selector.

    Select time series dimensions based on the sum of distances between pairs
    of classes.

    Parameters
    ----------
    prototype : {"mean", "median", "dtw"}, optional
        The method for computing the class prototypes.
    r : float, optional
        The warping width if `prototype` is "dtw".
    metric : str, optional
        The distance metric.
    metric_params : dict, optional
        Optional parameters to the distance metric.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_metrics>`.
    """

    _parameter_constraints = {
        "prototype": [StrOptions(_PROTOTYPE.keys())],
        "r": [
            None,
            Interval(numbers.Integral, 0, None, closed="left"),
            Interval(numbers.Real, 0, 1, closed="both"),
        ],
        "metric": [MetricOptions()],
        "metric_params": [None, dict],
    }

    def __init__(
        self, prototype="mean", r=None, metric="euclidean", metric_params=None
    ):
        self.prototype = prototype
        self.metric = metric
        self.metric_params = metric_params
        self.r = r

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Learn the dimensions to select.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep)
            The training samples.
        y : array-like of shape (n_samples, )
            Ignored.

        Returns
        -------
        object
            The instance itself.
        """
        X, y = self._validate_data(X, y, allow_3d=True, ensure_min_dims=2)
        check_classification_targets(y)
        labels = np.unique(y)
        n_labels = len(labels)
        n_combinations = int(n_labels * (n_labels - 1) / 2)
        _, n_dims, n_timestep = X.shape

        prototype_factory = _PROTOTYPE[self.prototype](
            self.r if self.r is not None else 1.0
        )
        prototypes = np.empty((n_dims, n_labels, n_timestep), dtype=X.dtype)
        for i in range(n_labels):
            for d in range(n_dims):
                prototypes[d, i] = prototype_factory.create(X[y == labels[i], d])

        proto_dist = np.empty((n_dims, n_combinations), dtype=float)
        for d in range(n_dims):
            k = 0
            for i in range(n_labels):
                for j in range(i + 1, n_labels):
                    proto_dist[d, k] = pairwise_distance(
                        prototypes[d, i], prototypes[d, j], metric=self.metric
                    )
                    k += 1

        dim_sum = np.sum(proto_dist, axis=1)
        self.dim_sort_ind_ = np.argsort(dim_sum)[::-1]
        p = np.linspace(dim_sum.max(), dim_sum.min(), dim_sum.size)
        self.elbow_ = np.abs(dim_sum[self.dim_sort_ind_] - p).argmax()
        return self

    def _get_dimensions(self):
        mask = np.zeros(self.n_dims_in_, dtype=bool)
        mask[self.dim_sort_ind_[: self.elbow_]] = True
        return mask
