# Authors: Isak Samsten
# License: BSD 3 clause

"""
Lower bounds for distance metrics.
"""

import numbers

import numpy as np
from sklearn.base import TransformerMixin, _fit_context, check_is_fitted
from sklearn.utils._param_validation import Interval, StrOptions

from ..base import BaseEstimator
from ..distance._distance import pairwise_distance
from ..transform._sax import PAA, SAX
from ._clb import pairwise_sax_distance
from ._elastic import _dtw_envelop, _dtw_lb_keogh
from .dtw import _compute_warp_size

__all__ = [
    "DtwKeoghLowerBound",
    "DtwKimLowerBound",
    "PaaLowerBound",
    "SaxLowerBound",
]


class SaxLowerBound(TransformerMixin, BaseEstimator):
    """
    Lower bound for the Euclidean distance between z-normalized time series.

    The lower bound is computed based on SAX.

    Parameters
    ----------
    window : int, optional
        The size of an interval. If `window`, is given then `n_intervals` is ignored.
    n_intervals : {"sqrt", "log2"}, int or float, optional
        The number of intervals.
    n_bins : int, optional
        The number of bins.

    Examples
    --------
    >>> from wildboar.datasets import load_gun_point
    >>> from wildboar.distance import argmin_distance
    >>> from wildboar.distance.lb import SaxLowerBound
    >>> X, y = load_gun_point()
    >>> lbsax = SaxLowerBound(n_bins=20).fit(X[30:])
    >>> argmin_distance(X[:30], X[30:], lower_bound=lbsax.transform(X[:30]))
    """

    _parameter_constraints: dict = {
        "n_intervals": [
            Interval(numbers.Integral, 1, None, closed="left"),
            Interval(numbers.Real, 0, 1, closed="right"),
            StrOptions({"sqrt", "log2"}),
        ],
        "window": [None, Interval(numbers.Integral, 1, None, closed="left")],
        "n_bins": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(self, *, window=None, n_intervals="sqrt", n_bins=10):
        self.window = window
        self.n_intervals = n_intervals
        self.n_bins = n_bins

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X):
        """
        Fit the lower bound for time series.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestep)
            The time series to query.

        Returns
        -------
        self
            The estimator.
        """
        X = self._validate_data(X)
        self.sax_ = SAX(
            n_intervals=self.n_intervals,
            window=self.window,
            n_bins=self.n_bins,
            scale=False,
        )
        self.X_ = self.sax_.fit_transform(X)
        return self

    def transform(self, X):
        """
        Compute lower bound for query.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_timesteps)
            The query.

        Returns
        -------
        ndarray of shape (n_queries, n_samples)
            The lower bound of the distance between the i:th
            query and the j:th in database sample.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        X = self.sax_.transform(X)
        return pairwise_sax_distance(
            X, self.X_, self.sax_.thresholds_, self.n_timesteps_in_
        )


class PaaLowerBound(TransformerMixin, BaseEstimator):
    """
    Lower bound for the Euclidean distance between z-normalized time series.

    The lower bound is computed based on PAA.

    Parameters
    ----------
    window : int, optional
        The size of an interval. If `window`, is given then `n_intervals` is ignored.
    n_intervals : {"sqrt", "log2"}, int or float, optional
        The number of intervals.

    Examples
    --------
    >>> from wildboar.datasets import load_gun_point
    >>> from wildboar.distance import argmin_distance
    >>> from wildboar.distance.lb import PaaLowerBound
    >>> X, y = load_gun_point()
    >>> lbpaa = PaaLowerBound().fit(X[30:])
    >>> argmin_distance(X[:30], X[30:], lower_bound=lbpaa.transform(X[:30]))
    """

    _parameter_constraints: dict = {
        "n_intervals": [
            Interval(numbers.Integral, 1, None, closed="left"),
            Interval(numbers.Real, 0, 1, closed="right"),
            StrOptions({"sqrt", "log2"}),
        ],
        "window": [None, Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(self, *, window=None, n_intervals="sqrt"):
        self.window = window
        self.n_intervals = n_intervals

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X):
        """
        Fit the lower bound for time series.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestep)
            The time series to query.

        Returns
        -------
        self
            The estimator.
        """
        X = self._validate_data(X)
        self.paa_ = PAA(n_intervals=self.n_intervals, window=self.window)
        self.X_ = self.paa_.fit_transform(X)
        return self

    def transform(self, X):
        """
        Compute lower bound for query.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_timesteps)
            The query.

        Returns
        -------
        ndarray of shape (n_queries, n_samples)
            The lower bound of the distance between the i:th
            query and the j:th in database sample.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        X = self.paa_.transform(X)
        return pairwise_distance(X, self.X_, dim=0, metric="euclidean")


class DtwKimLowerBound(TransformerMixin, BaseEstimator):
    """
    Lower bound for Dynamic time warping computed in constant time.

    The bound is very fast to compute but ineffective.
    """

    def fit(self, X):
        """
        Fit the lower bound for time series.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestep)
            The time series to query.

        Returns
        -------
        self
            The estimator.
        """
        self.X_ = self._validate_data(X)
        return self

    def transform(self, X):
        """
        Compute lower bound for query.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_timesteps)
            The query.

        Returns
        -------
        ndarray of shape (n_queries, n_samples)
            The lower bound of the distance between the i:th
            query and the j:th in database sample.
        """
        Y = self._validate_data(X, reset=False)

        def dist(x, y):
            v = x - y
            return v * v

        out = np.empty((Y.shape[0], self.X_.shape[0]), dtype=float)

        for i in range(Y.shape[0]):
            x0 = self.X_[:, 0]
            x0_ = self.X_[:, -1]
            y0 = Y[i, 0]
            y0_ = Y[i, -1]

            x1 = self.X_[:, 1]
            x1_ = self.X_[:, -2]
            y1 = Y[i, 1]
            y1_ = Y[i, -2]

            x2 = self.X_[:, 2]
            x2_ = self.X_[:, -3]
            y2 = Y[i, 2]
            y2_ = Y[i, -3]

            d = dist(x0, y0) + dist(x0_, y0_)
            d += np.minimum(
                dist(x1, y0),
                np.minimum(
                    dist(x0, y1),
                    dist(x1, y1),
                ),
            )
            d += np.minimum(
                dist(x1_, y1),
                np.minimum(
                    dist(x0_, y1_),
                    dist(x1_, y1_),
                ),
            )
            d += np.minimum(
                dist(x0, y2),
                np.minimum(
                    dist(x1, y2),
                    np.minimum(
                        dist(x2, y2),
                        np.minimum(
                            dist(x2, y1),
                            dist(x2, y0),
                        ),
                    ),
                ),
            )
            d += np.minimum(
                dist(x0_, y2_),
                np.minimum(
                    dist(x1_, y2_),
                    np.minimum(
                        dist(x2_, y2_),
                        np.minimum(
                            dist(x2_, y1_),
                            dist(x2_, y0_),
                        ),
                    ),
                ),
            )
            out[i, :] = d

        return out


class DtwKeoghLowerBound(TransformerMixin, BaseEstimator):
    """
    Lower bound for dynamic time warping.

    Implements the LB_Keogh algorithm for efficient similarity search in time series data.
    This method approximates distances between sequences by comparing their
    upper and lower bounds, enhancing performance by reducing computational
    overhead.

    Parameters
    ----------
    r : float
        The warp window for DTW.

    kind : {"both", "left", "right"}
        - If "both", compute the bound for both sides and take the maximum.
        - If "left", compute the bound only for the query.
        - If "right", compute the bound only for the data.

    Examples
    --------
    >>> from wildboar.datasets import load_gun_point
    >>> from wildboar.distance import argmin_distance
    >>> from wildboar.distance.lb import DtwKeoghLowerBound
    >>> X, y = load_gun_point()
    >>> lbkeogh = DtwKeoghLowerBound(r=0.1).fit(X[30:])
    >>> argmin_distance(
    ...     X[:30],  # query
    ...     X[30:],  # database
    ...     metric="dtw",
    ...     metric_params={"r": 0.1},
    ...     lower_bound=lbkeogh.transform(X[:30])
    ... )
    """

    _parameter_constraints = {
        "r": [Interval(numbers.Real, 0, 1, closed="both")],
        "kind": [StrOptions({"both", "left", "right"})],
    }

    def __init__(self, r=1.0, *, kind="both"):
        self.r = r
        self.kind = kind

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X):
        """
        Fit the lower bound for time series.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestep)
            The time series to query.

        Returns
        -------
        self
            The estimator.
        """
        self.X_ = self._validate_data(X)
        self.lower_ = np.empty((X.shape[0], X.shape[1]), dtype=float)
        self.upper_ = np.empty((X.shape[0], X.shape[1]), dtype=float)
        for i in range(X.shape[0]):
            lower, upper = _dtw_envelop(X[i, :], _compute_warp_size(X.shape[1], self.r))
            self.lower_[i] = lower
            self.upper_[i] = upper

        return self

    def transform(self, X):
        """
        Fit the lower bound for time series.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestep)
            The time series to query.

        Returns
        -------
        self
            The estimator.
        """
        check_is_fitted(self)
        Y = self._validate_data(X, reset=False)
        out = np.empty((Y.shape[0], self.X_.shape[0]), dtype=float)
        r = _compute_warp_size(Y.shape[1], self.r)
        compute_left = self.kind == "both" or self.kind == "left"
        compute_right = self.kind == "both" or self.kind == "right"
        for i in range(Y.shape[0]):
            if compute_right:
                lower, upper = _dtw_envelop(Y[i], r)

            for j in range(self.X_.shape[0]):
                if compute_left:
                    dist1, _ = _dtw_lb_keogh(Y[i], self.lower_[j], self.upper_[j], r)
                else:
                    dist1 = -np.inf

                if compute_right:
                    dist2, _ = _dtw_lb_keogh(self.X_[j], lower, upper, r)
                else:
                    dist2 = -np.inf

                out[i, j] = max(dist1, dist2)
        return out
