import numbers

import numpy as np
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from ._base import BaseDistanceSelector


class DistanceVarianceThreshold(BaseDistanceSelector):
    """
    Distance selector that removes dimensions with low variance.

    This dimension selector is suitable for unsupervised learning since it only
    considers the input data and not the labels.

    For each dimension, the pairwise distance between time series is computed
    and dimensions with variance below the specified threshold are removed.

    Parameters
    ----------
    threshold : float, optional
        The variance threshold.
    sample : int or float, optional
        Draw a sample of time series for the pairwise distance calculation.

        - If None, use all samples.
        - If float, use the specified fraction of samples.
        - If int, use the specified number of samples.
    metric : str, optional
        The distance metric.
    metric_params : dict, optional
        Optional parameters to the distance metric.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_metrics>`.
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

    Examples
    --------
    >>> from wildboar.datasets import load_ering
    >>> from wildboar.dimension_selection import DistanceVarianceThreshold
    >>> X, y = load_ering()
    >>> dv = DistanceVarianceThreshold(threshold=9)
    >>> dv.fit(X, y)
    DistanceVarianceThreshold(threshold=9)
    >>> dv.get_dimensions()
    array([ True, False,  True,  True])
    >>> dv.transform(X).shape
    """

    _parameter_constraints = {
        **BaseDistanceSelector._parameter_constraints,
        "threshold": [Interval(numbers.Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        threshold=0,
        *,
        sample=None,
        metric="euclidean",
        metric_params=None,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            sample=sample,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.threshold = threshold

    def _fit(self, dist, y=None):
        self.variances_ = np.var(dist, axis=2).mean(axis=1)
        if self.threshold == 0:
            self.variances_ = np.nanmin(
                [self.variances_, np.ptp(dist, axis=2).mean(axis=1)], axis=0
            )

        if np.all(~np.isfinite(self.variances_) | (self.variances_ <= self.threshold)):
            raise ValueError(
                f"No dimension is above the variance threshold {self.threshold}"
            )

        return self

    def _get_dimensions(self):
        check_is_fitted(self)
        return self.variances_ > self.threshold
