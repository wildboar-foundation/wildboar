# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten
import math

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
from .base import BaseEmbedding

_SUMMARIZER = {
    "auto": MeanVarianceSlopeSummarizer,
    "mean_var_slope": MeanVarianceSlopeSummarizer,
    "mean": MeanSummarizer,
    "variance": VarianceSummarizer,
    "slope": SlopeSummarizer,
    "catch22": Catch22Summarizer,
}


class IntervalEmbedding(BaseEmbedding):
    """Embed a time series as a collection of features per interval.

    Examples
    ========

    >>> from wildboar.datasets import load_dataset
    >>> x, y = load_dataset("GunPoint")
    >>> embedding = IntervalEmbedding(n_interval=10, summarizer="mean")
    >>> embedding.fit_transform(x)

    Each interval (15 timepoints) are embedded as its mean.

    >>> embedding = IntervalEmbedding(n_interval="sqrt", summarizer=[np.mean, np.std])
    >>> embedding.fit_transform(x)

    Each interval (150 // 12 timepoints) are embedded as two features. The mean
    and the standard deviation.

    """

    def __init__(
        self,
        n_interval="sqrt",
        *,
        intervals="fixed",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        summarizer="auto",
        n_jobs=None,
        random_state=None,
    ):
        """

        Parameters
        ----------
        n_interval : str, int or float, optional
            The number of intervals to use for the embedding.

            - if float, a fraction of n_timestep
            - if int, a fixed number of intervals
            - if "sqrt", sqrt(n_timestep)
            - if "log", log2(n_timestep)

        intervals : str, optional
            The method for selecting intervals

            - if "fixed", intervals are distributed evenly over the time series
              without overlaps

            - if "sample", a sample of non-overlapping intervals as selected
              by fixed are selected. The size of the sample is determined by
              `sample_size`.

            - if "random", a sample of possibly overlapping intervals. The size of
              the interval is determined by `min_size` and `max_size`

        sample_size : float, optional
            The sample size of fixed intervals if `intervals="sample"`

        min_size : float, optional
            The minimum interval size if `intervals="random"`

        max_size : float, optional
            The maximum interval size if `intervals="random"`

        summarizer : str or list, optional
            The method to summarize each interval.

            - if str, the summarizer is determined by `_SUMMARIZERS.keys()`.
            - if list, the summarizer is a list of functions f(x) -> float, where
              x is a numpy array.

            The default summarizer summarizes each interval as its mean, standard
            deviation and slope.

        n_jobs : int, optional
            The number of cores to use on multi-core.

        random_state : int or np.RandomState
            The pseudo-random number generator used to ensure consistent results.
        """
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.n_interval = n_interval
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

        if self.n_interval == "sqrt":
            n_interval = math.ceil(math.sqrt(self.n_timestep_))
        elif self.n_interval == "log":
            n_interval = math.ceil(math.log2(self.n_timestep_))
        elif isinstance(self.n_interval, int):
            n_interval = self.n_interval
        elif isinstance(self.n_interval, float):
            if not 0.0 < self.n_interval < 1.0:
                raise ValueError("n_interval must be between 0.0 and 1.0")
            n_interval = math.floor(self.n_interval * self.n_timestep_)
            # TODO: ensure that no interval is smaller than 2
        else:
            raise ValueError("n_interval (%r) is not supported" % self.n_interval)

        if self.intervals == "fixed":
            return IntervalFeatureEngineer(n_interval, summarizer)
        elif self.intervals == "sample":
            if not 0.0 < self.sample_size < 1.0:
                raise ValueError("sample_size must be between 0.0 and 1.0")

            sample_size = math.floor(n_interval * self.sample_size)
            return RandomFixedIntervalFeatureEngineer(
                n_interval, summarizer, sample_size
            )
        elif self.intervals == "random":
            if not 0.0 <= self.min_size < self.max_size:
                raise ValueError("min_size must be between 0.0 and max_size")
            if not self.min_size < self.max_size <= 1.0:
                raise ValueError("max_size must be between min_size and 1.0")

            min_size = int(self.min_size * self.n_timestep_)
            max_size = int(self.max_size * self.n_timestep_)
            if min_size < 2:
                min_size = 2

            return RandomIntervalFeatureEngineer(
                n_interval, summarizer, min_size, max_size
            )
        else:
            raise ValueError("intervals (%r) is unsupported." % self.intervals)


class FeatureEmbedding(IntervalEmbedding):
    """Embed a time series as a number of features"""

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
        Lubba, Carl H., Sarab S. Sethi, Philip Knaute, Simon R. Schultz, Ben D. Fulcher,
        and Nick S. Jones.
            catch22: Canonical time-series characteristics.
            Data Mining and Knowledge Discovery 33, no. 6 (2019): 1821-1852.
        """
        super(FeatureEmbedding).__init__(
            self, n_interval=1, summarizer=summarizer, n_jobs=n_jobs
        )
