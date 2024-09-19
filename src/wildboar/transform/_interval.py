# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers

from sklearn.base import TransformerMixin, _fit_context
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from wildboar.base import BaseEstimator

from ._base import BaseAttributeTransform
from ._cinterval import (
    Catch22Summarizer,
    DyadicIntervalAttributeGenerator,
    FixedIntervalAttributeGenerator,
    MeanSummarizer,
    MeanVarianceSlopeSummarizer,
    PyFuncSummarizer,
    QuantSummarizer,
    RandomFixedIntervalAttributeGenerator,
    RandomIntervalAttributeGenerator,
    SlopeSummarizer,
    VarianceSummarizer,
)
from ._diff import DiffTransform, FftTransform

_SUMMARIZER = {
    "mean_var_slope": MeanVarianceSlopeSummarizer,
    "mean": MeanSummarizer,
    "variance": VarianceSummarizer,
    "slope": SlopeSummarizer,
    "catch22": Catch22Summarizer,
    "quant": QuantSummarizer,
}


# noqa: H0002
class IntervalMixin:
    """
    Mixin for interval based estimators.

    It provides an implementation for the `_get_generator` method
    which supports interval based transformation.

    The implementing class must have the following properties:

    - `n_intervals`
    - `intervals`
    - `min_size`
    - `max_size`
    - `sample_size`
    - `summarizer`

    See :class:`transform.IntervalTransform` for information about the
    properties.
    """

    _parameter_constraints: dict = {
        "n_intervals": [
            Interval(numbers.Integral, 1, None, closed="left"),
            Interval(numbers.Real, 0, 1, closed="right"),
            StrOptions({"sqrt", "log2"}),
        ],
        "intervals": [
            StrOptions({"fixed", "random", "dyadic"}),
        ],
        "depth": [
            Interval(numbers.Integral, 0, None, closed="neither"),
            StrOptions({"auto"}),
            None,
        ],
        "min_size": [
            Interval(numbers.Real, 0, 1, closed="both"),
            None,
        ],
        "max_size": [
            Interval(numbers.Real, 0, 1, closed="both"),
            None,
        ],
        "sample_size": [None, Interval(numbers.Real, 0, 1, closed="both")],
        "summarizer": [StrOptions(_SUMMARIZER.keys()), list],
        "summarizer_params": [dict, None],
    }

    def _get_generator(self, x, y):  # noqa: PLR0912
        if isinstance(self.summarizer, list):
            if not all(callable(func) for func in self.summarizer):
                raise ValueError(
                    "summarizer must be list of callable or str, got %r"
                    % self.summarizer
                )
            summarizer = PyFuncSummarizer(self.summarizer)
        else:
            if (
                hasattr(self, "summarizer_params")
                and self.summarizer_params is not None
            ):
                summarizer_params = self.summarizer_params
            else:
                summarizer_params = {}
            summarizer = _SUMMARIZER[self.summarizer](**summarizer_params)

        if self.n_intervals == "sqrt":
            n_intervals = math.ceil(math.sqrt(self.n_timesteps_in_))
        elif self.n_intervals == "log2":
            n_intervals = math.ceil(math.log2(self.n_timesteps_in_))
        elif isinstance(self.n_intervals, numbers.Integral):
            n_intervals = self.n_intervals
        else:
            n_intervals = math.ceil(self.n_intervals * self.n_timesteps_in_)

        intervals = self.intervals
        if intervals == "fixed" and n_intervals > self.n_timesteps_in_:
            raise ValueError(
                "The number of intervals must be fewer than or equal "
                "to the number of timesteps."
            )

        if intervals == "fixed":
            if self.sample_size is None or self.sample_size == 1.0:
                return FixedIntervalAttributeGenerator(summarizer, n_intervals)
            else:
                return RandomFixedIntervalAttributeGenerator(
                    summarizer, n_intervals, math.floor(self.sample_size * n_intervals)
                )
        elif intervals == "dyadic":
            if self.depth is None or (
                isinstance(self.depth, str) and self.depth == "auto"
            ):
                depth = min(math.floor(math.log2(self.n_timesteps_in_)) + 1, 6)
            else:
                if 2 ** (self.depth - 1) > self.n_timesteps_in_:
                    raise ValueError(
                        "The depth must be less than {}, got {}".format(
                            math.floor(math.log2(self.n_timesteps_in_)) + 1, self.depth
                        )
                    )
                elif self.depth < 1:
                    raise ValueError(
                        "The depth must be larger than 1, got {self.depth}"
                    )
                depth = self.depth

            return DyadicIntervalAttributeGenerator(summarizer, depth)
        else:  # "random"
            min_size = self.min_size if self.min_size is not None else 0
            max_size = self.max_size if self.max_size is not None else 1

            if min_size > max_size:
                raise ValueError(
                    f"The min_size parameter of {type(self).__qualname__} "
                    "must be <= max_size."
                )

            min_size = int(self.min_size * self.n_timesteps_in_)
            max_size = int(self.max_size * self.n_timesteps_in_)
            if min_size < 2:
                if self.n_timesteps_in_ < 2:
                    min_size = 1
                else:
                    min_size = 2

            return RandomIntervalAttributeGenerator(
                summarizer, n_intervals, min_size, max_size
            )


class IntervalTransform(IntervalMixin, BaseAttributeTransform):
    """
    Embed a time series as a collection of features per interval.

    Parameters
    ----------
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
        - if "dyadic", `2**depth-1+2**depth-1-depth" intervals.
        - if "random", `n_intervals` possibly overlapping intervals of randomly
          sampled in `[min_size * n_timestep, max_size * n_timestep]`.

        Read more in the :ref:`User Guide <interval_intervals>`
    sample_size : float, optional
        The sample size of fixed intervals if `intervals="fixed"`.
    depth : int, optional
        The maximum depth for dyadic intervals if `intervals="dyadic"`.
    min_size : float, optional
        The minimum interval size if `intervals="random"`.
    max_size : float, optional
        The maximum interval size if `intervals="random"`.
    summarizer : str or list, optional
        The method to summarize each interval.

        - if str, the summarizer is determined by `_SUMMARIZERS.keys()`.
        - if list, the summarizer is a list of functions `f(x) -> float`, where
          `x` is a numpy array.

        The default summarizer summarizes each interval as its mean, standard
        deviation and slope.

        Read more in the :ref:`User Guide <interval_summarizers>`
    summarizer_params : dict, optional
        A dictionary of parameters to the summarizer.
    n_jobs : int, optional
        The number of cores to use on multi-core.
    random_state : int or RandomState, optional
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
          by `np.random`.

    Notes
    -----
    Parallelization depends on releasing the global interpreter lock (GIL). As
    such, custom functions as summarizers reduces the performance. Wildboar
    implements summarizers for taking the mean ("mean"), variance ("variance")
    and slope ("slope") as well as their combination ("mean_var_slope") and the
    full suite of `catch22` features ("catch22"). In the future, we will allow
    downstream projects to implement their own summarizers in Cython which will
    allow for releasing the GIL.

    References
    ----------
    Lubba, Carl H., Sarab S. Sethi, Philip Knaute, Simon R. Schultz, \
            Ben D. Fulcher, and Nick S. Jones.
        catch22: Canonical time-series characteristics.
        Data Mining and Knowledge Discovery 33, no. 6 (2019): 1821-1852.

    Examples
    --------
    >>> from wildboar.datasets import load_dataset
    >>> x, y = load_dataset("GunPoint")
    >>> t = IntervalTransform(n_intervals=10, summarizer="mean")
    >>> t.fit_transform(x)

    Each interval (15 timepoints) are transformed to their mean.

    >>> t = IntervalTransform(n_intervals="sqrt", summarizer=[np.mean, np.std])
    >>> t.fit_transform(x)

    Each interval (`150 // 12` timepoints) are transformed to two features. The
    mean and the standard deviation.
    """

    _parameter_constraints: dict = {
        **IntervalMixin._parameter_constraints,
        **BaseAttributeTransform._parameter_constraints,
    }

    def __init__(
        self,
        n_intervals="sqrt",
        *,
        intervals="fixed",
        sample_size=None,
        depth=None,
        min_size=0.0,
        max_size=1.0,
        summarizer="mean_var_slope",
        summarizer_params=None,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.n_intervals = n_intervals
        self.summarizer = summarizer
        self.summarizer_params = summarizer_params
        self.intervals = intervals
        self.sample_size = sample_size
        self.depth = depth
        self.min_size = min_size
        self.max_size = max_size


class FeatureTransform(IntervalTransform):
    """
    Transform a time series as a number of features.

    Parameters
    ----------
    summarizer : str or list, optional
        The method to summarize each interval.

        - if str, the summarizer is determined by `_SUMMARIZERS.keys()`.
        - if list, the summarizer is a list of functions f(x) -> float, where x
          is a numpy array.

        The default summarizer summarizes each time series using `catch22`-features.
    n_jobs : int, optional
        The number of cores to use on multi-core.

    Examples
    --------
    >>> from wildboar.datasets import load_gun_point
    >>> X, y = load_gun_point()
    >>> X_t = FeatureTransform().fit_transform(X)
    >>> X_t[0]
    array([-5.19633603e-01, -6.51047206e-01,  1.90000000e+01,  4.80000000e+01,
            7.48441896e-01, -2.73293560e-05,  2.21476510e-01,  4.70000000e+01,
            4.00000000e-02,  0.00000000e+00,  2.70502518e+00,  2.60000000e+01,
            6.42857143e-01,  1.00000000e-01, -3.26666667e-01,  9.89974643e-01,
            2.90000000e+01,  1.31570726e+00,  1.50000000e-01,  8.50000000e-01,
            4.90873852e-02,  1.47311800e-01])
    """

    _parameter_constraints: dict = {
        "summarizer": IntervalMixin._parameter_constraints["summarizer"],
        "n_jobs": [None, numbers.Integral],
    }

    def __init__(
        self,
        *,
        summarizer="catch22",
        n_jobs=None,
    ):
        super().__init__(n_intervals=1, summarizer=summarizer, n_jobs=n_jobs)


class QuantTransform(TransformerMixin, BaseEstimator):
    """
    Quant transformation

    Computes quantiles over a fixed set of intervals on input time series
    and their transformations, using these quantiles for classification.

    The Quant transform performs the following steps:

    1. Computes quantiles over fixed, dyadic intervals on the input time
       series.
    2. Applies three transformations to the time series (first difference,
       second difference, and Fourier transform).

    Parameters
    ----------
    depth : {"auto"} or int, optional
        The maximal depth. If set to auto, the depth is `min(log2(n_timestep) +
        1, 6)`.
    v : int, optional
        The proportion of quantiles per interval given as `k = m/v` were `m` is
        the length of the interval.
    n_jobs : int, optional
        The number of parallel jobs.

    Notes
    -----
    The implementation differs to the original in the following ways:

    1. Does not apply smoothing to the first order difference.
    2. Does not subtract the mean from every second quantile.
    3. Does not apply 1-order differences if the time series are shorter than 2
       timesteps.
    4. Does not apply 2-order differences if the time series are shorter than 3
       timesteps.

    References
    ----------
    Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb.
        "Quant: A Minimalist Interval Method for Time Series Classification."
        Data Mining and Knowledge Discovery 38, no. 4 (July 1, 2024): 2377–2402.
        https://doi.org/10.1007/s10618-024-01036-9.
    """

    _parameter_constraints = {
        "depth": IntervalTransform._parameter_constraints["depth"],
        "v": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_jobs": [None, numbers.Integral],
    }

    def __init__(self, depth="auto", v=4, n_jobs=None):
        self.depth = depth
        self.v = v
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep)
            The input data.
        y : ignored

        Returns
        -------
        self
            The fitted estimator.
        """
        X = self._validate_data(X, allow_3d=True)
        params = dict(
            intervals="dyadic",
            depth=self.depth,
            summarizer="quant",
            summarizer_params={"v": self.v},
        )
        steps = [
            IntervalTransform(**params),
            make_pipeline(FftTransform(), IntervalTransform(**params)),
        ]
        if X.shape[-1] > 1:
            steps.append(
                make_pipeline(DiffTransform(order=1), IntervalTransform(**params))
            )

        if X.shape[-1] > 2:
            steps.append(
                make_pipeline(DiffTransform(order=2), IntervalTransform(**params))
            )

        self.pipe_ = make_union(*steps, n_jobs=self.n_jobs)
        self.pipe_.fit(X)
        return self

    def transform(self, X):
        """
        Transform the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep)
            The input data.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs)
            The transformed samples.
        """
        X = self._validate_data(X, reset=False, allow_3d=True)
        check_is_fitted(self)
        return self.pipe_.transform(X)
