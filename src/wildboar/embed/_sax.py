import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin

from wildboar.utils import check_array

from . import IntervalEmbedding


class SAX(TransformerMixin, BaseEstimator):
    """Symbolic aggregate approximation"""

    def __init__(self, *, n_intervals="sqrt", window=None, n_bins=4):
        """
        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep)
            The input data.

        n_intervals : str, optional
            The number of intervals to use for the embedding.

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
        """
        self.n_intervals = n_intervals
        self.window = window
        self.n_bins = n_bins

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return symbolic_aggregate_approximation(
            x, n_intervals=self.n_intervals, window=self.window, n_bins=self.n_bins
        )

    def __sklearn_is_fitted__(self):
        """Return True since FunctionTransfomer is stateless."""
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
        return piecewice_aggregate_approximation(
            x, n_intervals=self.n_intervals, window=self.window
        )

    def __sklearn_is_fitted__(self):
        """Return True since FunctionTransfomer is stateless."""
        return True

    def _more_tags(self):
        return {
            "requires_fit": False,
            "stateless": True,
            "no_validation": True,
        }


def symbolic_aggregate_approximation(x, *, n_intervals="sqrt", window=None, n_bins=4):
    """Symbolic aggregate approximation

    Parameters
    ----------
    x : array-like of shape (n_samples, n_timestep)
        The input data.

    n_intervals : str, optional
        The number of intervals to use for the embedding.

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

    Returns
    -------
    sax : ndarray of shape (n_samples, n_intervals)
        The symbolic aggregate approximation

    """
    x = check_array(x, dtype=float)
    x_paa = piecewice_aggregate_approximation(x, n_intervals=n_intervals, window=window)
    loc = np.mean(x, axis=1).reshape(-1, 1)
    scale = np.std(x, axis=1).reshape(-1, 1)
    percentiles = np.linspace(0, 1, num=n_bins, endpoint=False)[1:].reshape(1, -1)
    bins = norm.ppf(percentiles, loc=loc, scale=scale)
    x_out = np.empty(x_paa.shape, dtype=np.min_scalar_type(n_bins))
    for i, (x_i, bin_i) in enumerate(zip(x_paa, bins)):
        x_out[i] = np.digitize(x_i, bin_i)
    return x_out


def piecewice_aggregate_approximation(x, *, n_intervals="sqrt", window=None):
    """Peicewise aggregate approximation

    Parameters
    ----------
    x : array-like of shape (n_samples, n_timestep)
        The input data.

    n_intervals : str, optional
        The number of intervals to use for the embedding.

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
    if window is not None:
        if not 0 < window <= x.shape[-1]:
            raise ValueError("invalid window size, got %d" % window)
        n_intervals = x.shape[-1] // window
    x = check_array(x, dtype=float)
    return IntervalEmbedding(n_intervals=n_intervals, summarizer="mean").fit_transform(
        x
    )
