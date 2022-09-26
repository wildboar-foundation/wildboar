import numpy as np
from scipy.stats import norm, uniform
from sklearn.base import TransformerMixin

from ..base import BaseEstimator
from . import IntervalTransform


def _percentiles(n_bins):
    return np.linspace(0, 1, num=n_bins, endpoint=False)[1:].reshape(1, -1)


def _normal_bins(percentiles, x, estimate=True):
    if estimate:
        loc = np.mean(x, axis=1).reshape(-1, 1)
        scale = np.std(x, axis=1).reshape(-1, 1)
        return norm.ppf(percentiles, loc=loc, scale=scale)
    else:
        return np.repeat(norm.ppf(percentiles).reshape(1, -1), x.shape[0], axis=0)


def _uniform_bins(percentiles, x, estimate=True):
    if estimate:
        loc = np.min(x, axis=1).reshape(-1, 1)
        scale = np.max(x, axis=1).reshape(-1, 1) - loc
        return uniform.ppf(percentiles, loc=loc, scale=scale)
    else:
        return np.repeat(uniform.ppf(percentiles).reshape(1, -1), x.shape[0], axis=0)


_BINNING = {"normal": _normal_bins, "uniform": _uniform_bins}


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
        return self

    def transform(self, x, y=None):
        x = self._validate_data(x, dtype=float, reset=False)
        x_paa = PAA(n_intervals=self.n_intervals, window=self.window).fit_transform(x)
        if self.binning not in _BINNING.keys():
            raise ValueError("binning (%s) not supported." % self.binning)

        bins = _BINNING[self.binning](
            _percentiles(self.n_bins), x, estimate=self.estimate
        )
        x_out = np.empty(x_paa.shape, dtype=np.min_scalar_type(self.n_bins))
        for i, (x_i, bin_i) in enumerate(zip(x_paa, bins)):
            x_out[i] = np.digitize(x_i, bin_i)
        return x_out

    def __sklearn_is_fitted__(self):
        """Return True since SAX is stateless."""
        return True

    def _more_tags(self):
        return {
            "requires_fit": False,
            "stateless": True,
            "no_validation": True,
            "preserves_dtype": [],
        }


class PAA(TransformerMixin, BaseEstimator):
    """Peicewise aggregate approximation"""

    def __init__(self, n_intervals="sqrt", window=None):
        self.n_intervals = n_intervals
        self.window = window

    def fit(self, x=None, y=None):
        return self

    def transform(self, x, y=None):
        x = self._validate_data(x, dtype=float, reset=False)
        if self.window is not None:
            if not 0 < self.window <= x.shape[-1]:
                raise ValueError("invalid window size, got %d" % self.window)
            n_intervals = x.shape[-1] // self.window
        else:
            n_intervals = self.n_intervals

        return IntervalTransform(
            n_intervals=n_intervals, summarizer="mean"
        ).fit_transform(x)

    def __sklearn_is_fitted__(self):
        """Return True since SAX is stateless."""
        return True

    def _more_tags(self):
        return {
            "requires_fit": False,
            "stateless": True,
            "no_validation": True,
        }


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
