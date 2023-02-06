# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers
import warnings

import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions

from ..distance import _SUBSEQUENCE_METRICS
from ..distance._multi_metric import make_subsequence_metrics
from ._cshapelet import (
    RandomMultiMetricShapeletFeatureEngineer,
    RandomShapeletFeatureEngineer,
)
from .base import BaseFeatureEngineerTransform


class ShapeletMixin:
    _parameter_constraints: dict = {
        "n_shapelets": [
            Interval(numbers.Integral, 1, None, closed="left"),
            StrOptions({"log2", "sqrt", "warn"}),
            callable,
        ],
        "metric": [
            StrOptions(_SUBSEQUENCE_METRICS.keys()),
            list,
        ],
        "metric_params": [dict, None],
        "min_shapelet_size": [
            Interval(numbers.Real, 0, 1, closed="both"),
        ],
        "max_shapelet_size": [
            Interval(numbers.Real, 0, 1, closed="both"),
        ],
    }

    def _get_feature_engineer(self, n_samples):
        if self.min_shapelet_size > self.max_shapelet_size:
            raise ValueError(
                f"The min_shapelet_size parameter of {type(self).__qualname__} "
                "must be <= max_shapelet_size."
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
            else:
                n_shapelets = int(self.n_shapelets(possible_shapelets))
        else:
            n_shapelets = self.n_shapelets

        if isinstance(self.metric, str):
            metric_params = self.metric_params if self.metric_params is not None else {}
            return RandomShapeletFeatureEngineer(
                _SUBSEQUENCE_METRICS[self.metric](**metric_params),
                min_shapelet_size,
                max_shapelet_size,
                max(1, n_shapelets),
            )
        else:
            metrics, weights = make_subsequence_metrics(self.metric)
            return RandomMultiMetricShapeletFeatureEngineer(
                max(1, n_shapelets),
                min_shapelet_size,
                max_shapelet_size,
                metrics,
                weights,
            )


class RandomShapeletTransform(ShapeletMixin, BaseFeatureEngineerTransform):
    """Transform a time series to the distances to a selection of random shapelets.

    Attributes
    ----------
    embedding_ : Embedding
        The underlying embedding object.


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

    References
    ----------
    Wistuba, Martin, Josif Grabocka, and Lars Schmidt-Thieme.
        Ultra-fast shapelets for time series classification. arXiv preprint
        arXiv:1503.05018 (2015).
    """

    _parameter_constraints: dict = {
        **ShapeletMixin._parameter_constraints,
        **BaseFeatureEngineerTransform._parameter_constraints,
    }

    def __init__(
        self,
        n_shapelets=1000,
        *,
        metric="euclidean",
        metric_params=None,
        min_shapelet_size=0,
        max_shapelet_size=1.0,
        n_jobs=None,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_shapelets : int, optional
            The number of shapelets in the resulting transform

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

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.n_shapelets = n_shapelets
        self.metric = metric
        self.metric_params = metric_params
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
