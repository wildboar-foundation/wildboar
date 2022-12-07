# Authors: Isak Samsten
# License: BSD 3 clause

import abc
import numbers

import numpy as np
from scipy.stats import norm, uniform
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_scalar

from wildboar.utils.validation import check_array, check_option

from ..base import BaseEstimator
from . import IntervalTransform


def _percentiles(n_bins):
    return np.linspace(0, 1, num=n_bins, endpoint=False)[1:].reshape(1, -1)


class Binning(metaclass=abc.ABCMeta):
    def __init__(self, n_bins) -> None:
        self.n_bins = n_bins
        self._percentiles = _percentiles(n_bins)

    @abc.abstractmethod
    def scale(self, x):
        pass

    def get_thresholds(self, x=None, estimate=False):
        if estimate and x is None:
            raise ValueError("if estimate=True, x cannot be None.")

        return self._get_thresholds(x, estimate)

    @abc.abstractmethod
    def _get_thresholds(self, x):
        pass


class NormalBinning(Binning):
    def scale(self, x):
        # Avoid circular import
        from ..datasets.preprocess import standardize

        return standardize(x)

    def _get_thresholds(self, x, estimate):
        if estimate and x is not None:
            loc = np.mean(x, axis=1).reshape(-1, 1)
            scale = np.std(x, axis=1).reshape(-1, 1)
            return norm.ppf(self._percentiles, loc=loc, scale=scale)
        else:
            ppf = norm.ppf(self._percentiles)
            if x is None:
                return ppf.reshape(-1)
            else:
                return np.repeat(ppf, x.shape[0], axis=0)


class UniformBinning(Binning):
    def scale(self, x):
        # Avoid circular import
        from ..datasets.preprocess import minmax_scale

        return minmax_scale(x)

    def _get_thresholds(self, x, estimate):
        if estimate and x is not None:
            loc = np.min(x, axis=1).reshape(-1, 1)
            scale = np.max(x, axis=1).reshape(-1, 1) - loc
            return uniform.ppf(self._percentiles, loc=loc, scale=scale)
        else:
            ppf = uniform.ppf(self._percentiles)
            if x is None:
                return ppf.reshape(-1)
            else:
                return np.repeat(ppf, x.shape[0], axis=0)


_BINNING = {"normal": NormalBinning, "uniform": UniformBinning}


class SAX(TransformerMixin, BaseEstimator):
    """Symbolic aggregate approximation"""

    def __init__(
        self,
        *,
        n_intervals="sqrt",
        window=None,
        n_bins=4,
        binning="normal",
        estimate=True,
    ):
        """
        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep)
            The input data.

        n_intervals : str, optional
            The number of intervals to use for the transform.

            - if "log", the number of intervals is ``log2(n_timestep)``.
            - if "sqrt", the number of intervals is ``sqrt(n_timestep)``.
            - if int, the number of intervals is ``n_intervals``.
            - if float, the number of intervals is ``n_intervals * n_timestep``, with
                ``0 < n_intervals < 1``.

        window : int, optional
            The window size. If ``window`` is set, the value of ``n_intervals`` has no
            effect.

        n_bins : int, optional
            The number of bins.

        binning : str, optional
            The bin construction. By default the bins are defined according to the
            normal distribution. Possible values are ``"normal"`` for normally
            distributed bins or ``"uniform"`` for uniformly distributed bins.

        estimate : bool, optional
            Estimate the distribution parameters for the binning from data.

            If ``estimate=False``, it is assumed that each time series are:

            - preprocessed using :func:`datasets.preprocess.normalize` when
              ``binning="normal"``.

            - preprocessed using :func:`datasets.preprocess.minmax_scale`. when
              ``binning="uniform"``

        """
        self.n_intervals = n_intervals
        self.window = window
        self.n_bins = n_bins
        self.binning = binning
        self.estimate = estimate

    def fit(self, x, y=None):
        x = self._validate_data(x, dtype=float)
        self.binning_ = check_option(_BINNING, self.binning, "binning")(self.n_bins)
        self.bins_ = np.arange(self.n_bins, dtype=np.min_scalar_type(self.n_bins))
        self.paa_ = PAA(n_intervals=self.n_intervals, window=self.window).fit(x)
        return self

    def transform(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, reset=False, dtype=float)
        x_paa = self.paa_.transform(x)
        thresholds = self.binning_.get_thresholds(x, estimate=self.estimate)
        x_out = np.empty(x_paa.shape, dtype=np.min_scalar_type(self.n_bins))
        for i, (sample, threshold) in enumerate(zip(x_paa, thresholds)):
            x_out[i] = np.digitize(sample, threshold)
        return x_out

    def inverse_transform(self, x):
        check_is_fitted(self)
        if self.estimate:
            raise ValueError("Unable to inverse_transform with estimate=True")

        x = check_array(x, dtype=np.min_scalar_type(self.n_bins))
        thresholds = self.binning_.get_thresholds()
        thresholds = np.hstack(
            [thresholds[0], (thresholds[:-1] + thresholds[1:]) / 2, thresholds[-1]]
        )

        x_inverse = np.empty((x.shape[0], self.n_timesteps_in_), dtype=float)
        for i, (start, end) in enumerate(self.intervals):
            x_inverse[:, start:end] = thresholds[x[:, i], np.newaxis]

        return x_inverse

    @property
    def intervals(self):
        return self.paa_.intervals

    def _more_tags(self):
        return {
            "preserves_dtype": [],
        }


class PAA(TransformerMixin, BaseEstimator):
    """Peicewise aggregate approximation"""

    def __init__(self, n_intervals="sqrt", window=None):
        self.n_intervals = n_intervals
        self.window = window

    def fit(self, x, y=None):
        x = self._validate_data(x, dtype=float)
        if self.window is not None:
            n_intervals = x.shape[-1] // check_scalar(
                self.window,
                "window",
                numbers.Integral,
                min_val=1,
                max_val=self.n_timesteps_in_,
            )
        else:
            n_intervals = self.n_intervals

        self.interval_transform_ = IntervalTransform(
            n_intervals=n_intervals, summarizer="mean"
        )
        self.interval_transform_.fit(x)
        return self

    def transform(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, reset=False)
        return self.interval_transform_.transform(x)

    def inverse_transform(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=float)

        x_inverse = np.empty((x.shape[0], self.n_timesteps_in_), dtype=float)
        for i, (start, end) in enumerate(self.intervals):
            x_inverse[:, start:end] = x[:, i].reshape(-1, 1)

        return x_inverse

    @property
    def intervals(self):
        return [
            (start, start + length)
            for (_, (start, length, _)) in self.interval_transform_.embedding_.features
        ]


def symbolic_aggregate_approximation(
    x,
    *,
    n_intervals="sqrt",
    window=None,
    n_bins=4,
    binning="normal",
):
    """Symbolic aggregate approximation

    Parameters
    ----------
    x : array-like of shape (n_samples, n_timestep)
        The input data.

    n_intervals : str, optional
        The number of intervals to use for the transform.

        - if "log", the number of intervals is ``log2(n_timestep)``.
        - if "sqrt", the number of intervals is ``sqrt(n_timestep)``.
        - if int, the number of intervals is ``n_intervals``.
        - if float, the number of intervals is ``n_intervals * n_timestep``, with
            ``0 < n_intervals < 1``.

    window : int, optional
        The window size. If ``window`` is set, the value of ``n_intervals`` has no
        effect.

    n_bins : int, optional
        The number of bins.

    binning : str, optional
        The bin construction. By default the bins are defined according to the
        normal distribution. Possible values are ``"normal"`` for normally
        distributed bins or ``"uniform"`` for uniformly distributed bins.

    Returns
    -------
    sax : ndarray of shape (n_samples, n_intervals)
        The symbolic aggregate approximation

    """
    return SAX(
        n_intervals=n_intervals, window=window, n_bins=n_bins, binning=binning
    ).fit_transform(x)


def piecewice_aggregate_approximation(x, *, n_intervals="sqrt", window=None):
    """Peicewise aggregate approximation

    Parameters
    ----------
    x : array-like of shape (n_samples, n_timestep)
        The input data.

    n_intervals : str, optional
        The number of intervals to use for the transform.

        - if "log", the number of intervals is ``log2(n_timestep)``.
        - if "sqrt", the number of intervals is ``sqrt(n_timestep)``.
        - if int, the number of intervals is ``n_intervals``.
        - if float, the number of intervals is ``n_intervals * n_timestep``, with
            ``0 < n_intervals < 1``.

    window : int, optional
        The window size. If ``window`` is set, the value of ``n_intervals`` has no
        effect.

    Returns
    -------
    paa : ndarray of shape (n_samples, n_intervals)
        The symbolic aggregate approximation

    """
    return PAA(n_intervals=n_intervals, window=window).fit_transform(x)
