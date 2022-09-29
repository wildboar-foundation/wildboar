# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers

from ._cinterval import (
    Catch22Summarizer,
    IntervalFeatureEngineer,
    MeanSummarizer,
    MeanVarianceSlopeSummarizer,
    PyFuncSummarizer,
    RandomFixedIntervalFeatureEngineer,
    RandomIntervalFeatureEngineer,
    SlopeSummarizer,
    VarianceSummarizer,
)
from .base import BaseFeatureEngineerTransform

_SUMMARIZER = {
    "mean_var_slope": MeanVarianceSlopeSummarizer,
    "mean": MeanSummarizer,
    "variance": VarianceSummarizer,
    "slope": SlopeSummarizer,
    "catch22": Catch22Summarizer,
}


class IntervalTransform(BaseFeatureEngineerTransform):
    """Embed a time series as a collection of features per interval.

    Examples
    --------

    >>> from wildboar.datasets import load_dataset
    >>> x, y = load_dataset("GunPoint")
    >>> t = IntervalTransform(n_intervals=10, summarizer="mean")
    >>> t.fit_transform(x)

    Each interval (15 timepoints) are transformed to their mean.

    >>> t = IntervalTransform(n_intervals="sqrt", summarizer=[np.mean, np.std])
    >>> t.fit_transform(x)

    Each interval (150 // 12 timepoints) are transformed to two features. The mean
    and the standard deviation.

    """

    def __init__(
        self,
        n_intervals="sqrt",
        *,
        intervals="fixed",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        summarizer="mean_var_slope",
        n_jobs=None,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_intervals : str, int or float, optional
            The number of intervals to use for the transform.

            - if "log", the number of intervals is ``log2(n_timestep)``.
            - if "sqrt", the number of intervals is ``sqrt(n_timestep)``.
            - if int, the number of intervals is ``n_intervals``.
            - if float, the number of intervals is ``n_intervals * n_timestep``, with
              ``0 < n_intervals < 1``.

        intervals : str, optional
            The method for selecting intervals

            - if "fixed", `n_intervals` non-overlapping intervals.
            - if "sample", ``n_intervals * sample_size`` non-overlapping intervals.
            - if "random", `n_intervals` possibly overlapping intervals of randomly
              sampled in ``[min_size * n_timestep, max_size * n_timestep]``

        sample_size : float, optional
            The sample size of fixed intervals if ``intervals="sample"``

        min_size : float, optional
            The minimum interval size if ``intervals="random"``

        max_size : float, optional
            The maximum interval size if ``intervals="random"``

        summarizer : str or list, optional
            The method to summarize each interval.

            - if str, the summarizer is determined by ``_SUMMARIZERS.keys()``.
            - if list, the summarizer is a list of functions ``f(x) -> float``, where
              x is a numpy array.

            The default summarizer summarizes each interval as its mean, standard
            deviation and slope.

        n_jobs : int, optional
            The number of cores to use on multi-core.

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.n_intervals = n_intervals
        self.summarizer = summarizer
        self.intervals = intervals
        self.sample_size = sample_size
        self.min_size = min_size
        self.max_size = max_size

    def _get_feature_engineer(self):
        if isinstance(self.summarizer, list):
            if not all(hasattr(func, "__call__") for func in self.summarizer):
                raise ValueError("summarizer (%r) is not supported")
            summarizer = PyFuncSummarizer(self.summarizer)
        else:
            summarizer = _SUMMARIZER.get(self.summarizer)()
            if summarizer is None:
                raise ValueError("summarizer (%r) is not supported." % self.summarizer)

        if self.n_intervals == "sqrt":
            n_intervals = math.ceil(math.sqrt(self.n_timesteps_in_))
        elif self.n_intervals == "log":
            n_intervals = math.ceil(math.log2(self.n_timesteps_in_))
        elif isinstance(self.n_intervals, numbers.Integral):
            if not 0 < self.n_intervals <= self.n_timesteps_in_:
                raise ValueError(
                    "n_intervals must be in the range [1, x.shape[-1]], got %d"
                    % self.n_intervals,
                )
            n_intervals = self.n_intervals
        elif isinstance(self.n_intervals, numbers.Real):
            if not 0.0 < self.n_intervals < 1.0:
                raise ValueError("n_intervals must be between 0.0 and 1.0")
            n_intervals = math.max(
                1, math.floor(self.n_intervals * self.n_timesteps_in_)
            )
        else:
            raise ValueError("n_intervals (%r) is not supported" % self.n_intervals)

        if self.intervals == "fixed":
            return IntervalFeatureEngineer(n_intervals, summarizer)
        elif self.intervals == "sample":
            if not 0.0 < self.sample_size < 1.0:
                raise ValueError("sample_size must be between 0.0 and 1.0")

            sample_size = math.max(1, math.floor(n_intervals * self.sample_size))
            return RandomFixedIntervalFeatureEngineer(
                n_intervals, summarizer, sample_size
            )
        elif self.intervals == "random":
            if not 0.0 <= self.min_size < self.max_size:
                raise ValueError("min_size must be between 0.0 and max_size")
            if not self.min_size < self.max_size <= 1.0:
                raise ValueError("max_size must be between min_size and 1.0")

            min_size = int(self.min_size * self.n_timesteps_in_)
            max_size = int(self.max_size * self.n_timesteps_in_)
            if min_size < 2:
                min_size = 2

            return RandomIntervalFeatureEngineer(
                n_intervals, summarizer, min_size, max_size
            )
        else:
            raise ValueError("intervals (%r) is unsupported." % self.intervals)


class FeatureTransform(IntervalTransform):
    """Transform a time series as a number of features"""

    def __init__(
        self,
        *,
        summarizer="catch22",
        n_jobs=None,
    ):
        """
        Parameters
        ----------
        summarizer : str or list, optional
            The method to summarize each interval.

            - if str, the summarizer is determined by `_SUMMARIZERS.keys()`.

            - if list, the summarizer is a list of functions f(x) -> float, where
              x is a numpy array.

            The default summarizer summarizes each time series using catch22-features

        n_jobs : int, optional
            The number of cores to use on multi-core.

        References
        ==========
        Lubba, Carl H., Sarab S. Sethi, Philip Knaute, Simon R. Schultz, \
        Ben D. Fulcher, and Nick S. Jones.
            catch22: Canonical time-series characteristics.
            Data Mining and Knowledge Discovery 33, no. 6 (2019): 1821-1852.
        """
        super(FeatureTransform).__init__(
            self, n_intervals=1, summarizer=summarizer, n_jobs=n_jobs
        )
