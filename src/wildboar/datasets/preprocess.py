# Authors: Isak Samsten
# License: BSD 3 clause
"""Utilities for preprocessing time series."""

import numbers
from functools import partial

import numpy as np
from scipy import version as scipy_version
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator
from sklearn.base import OneToOneFeatureMixin, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator
from ..transform._sax import piecewice_aggregate_approximation
from ..utils.validation import check_array, check_option
from ..utils.variable_len import is_end_of_series

__SCIPY_VERSION = tuple(int(v) for v in scipy_version.version.split("."))
__all__ = [
    "standardize",
    "minmax_scale",
    "maxabs_scale",
    "truncate",
    "named_preprocess",
    "interpolate",
    "MinMaxScale",
    "MaxAbsScale",
    "Standardize",
    "Interpolate",
    "Truncate",
]


def named_preprocess(name):
    """
    Get a named preprocessor.

    Parameters
    ----------
    name : str
        The name of the preprocessor.

    Returns
    -------
    callable
        The preprocessor function.
    """
    return check_option(_PREPROCESS, name, "name")


def standardize(x):
    """
    Scale x along the time dimension.

    The resulting array will have zero mean and unit standard deviation.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The samples.

    Returns
    -------
    ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The standardized samples.
    """
    return Standardize().fit_transform(x)


def minmax_scale(x, min=0, max=1):
    """
    Scale x along the time dimension.

    Each time series is scaled such that each value is between min and max.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The samples.
    min : float, optional
        The minimum value.
    max : float, optional
        The maximum value.

    Returns
    -------
    ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The transformed samples.
    """
    return MinMaxScale(min=min, max=max).fit_transform(x)


def maxabs_scale(x):
    """
    Scale each time series by its maximum absolute value.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The samples.

    Returns
    -------
    ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The transformed samples.
    """
    return MaxAbsScale().fit_transform(x)


def truncate(x):
    """
    Truncate x to the shortest sequence.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The samples.

    Returns
    -------
    ndarray of shape (n_samples, n_shortest) or (n_samples, n_dims, n_shortest)
        The truncated samples.
    """
    return Truncate().fit_transform(x)


# Wrapper for np.interp to match the scipy.interpolate interface.
class __NpInterp:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, i):
        return np.interp(i, self.x, self.y)


_INTERPOLATE_METHOD = {
    "linear": __NpInterp,
    "cubic": CubicSpline,
    "pchip": partial(PchipInterpolator, extrapolate=True),
}

if __SCIPY_VERSION >= (1, 13, 0):
    _INTERPOLATE_METHOD["akima"] = partial(Akima1DInterpolator, extrapolate=True)
    _INTERPOLATE_METHOD["makima"] = partial(
        Akima1DInterpolator, method="makima", extrapolate=True
    )


def interpolate(X, method="linear"):
    """
    Interpolate the given time series using the specified method.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
        The input data to be interpolated. It can be of any shape but must
        have at least dimension.
    method : str, optional
        The interpolation method to use. Default is "linear".

    Notes
    -----
    If scipy < 1.4, valid `method` values include "linear", "pchip", and
    "cubic". Otherwise, `method` also supports "akima" and "makima".

    Returns
    -------
    ndarray
        The interpolated data.
    """
    return Interpolate(method=method).fit_transform(X)


class SparseScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    Scale attributes.

    The attributes are scaled by:::

        sqrt(x) - mean(sqrt(x), axis=0) / std(sqrt(x), axis=0) + epsilon

    where `epsilon` is given by ``mean(sqrt(x) == 0, axis=0) ** exp``

    Parameters
    ----------
    mask_zero : bool, optional
        Keep zero values at zero.
    exp : float, optional
        The exponent of the feature sparcity.
    """

    _parameter_constraints: dict = {
        "mask_zero": [bool],
        "exp": Interval(numbers.Real, 0, None, closed="left"),
    }

    def __init__(self, mask_zero=True, exp=4):
        self.exp = exp
        self.mask_zero = mask_zero

    def fit(self, x, y=None):
        """
        Fit the model using x (ignores y)

        Parameters
        ----------
        x : array-like
            Training data.
        : array-like, optional
            Target values (default is None).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        x = np.sqrt(self._validate_data(x, allow_3d=False).clip(min=0))

        self.mu_ = x.mean(axis=0)
        self.sigma_ = x.std(axis=0) + (x == 0).mean() ** self.exp + 1e-8
        return self

    def transform(self, x):
        """
        Transform the input data using the stored parameters.

        Parameters
        ----------
        x : array-like
            Input data to be transformed.

        Returns
        -------
        array-like
            Transformed data.
        """
        x = np.sqrt(self._validate_data(x, allow_3d=False).clip(min=0))

        x = x - self.mu_
        if self.mask_zero:
            x *= x != 0

        x /= self.sigma_

        return x


class Interpolate(TransformerMixin, BaseEstimator):
    """
    Interpolate missing (`np.nan`) values.

    Parameters
    ----------
    method : str, optional
        The interpolation method to use. Default is "linear".

    Notes
    -----
    If scipy < 1.4, valid `method` values include "linear", "pchip", and
    "cubic". Otherwise, `method` also supports "akima" and "makima".
    """

    _parameter_constraints: dict = {
        "method": [StrOptions(_INTERPOLATE_METHOD.keys())],
    }

    def __init__(self, method="linear"):
        self.method = method

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the model to the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The input data to fit the model.
        y : array-like, optional
            The target values. Ignored.

        Returns
        -------
        object
            Returns the instance of the fitted model.
        """
        self._validate_data(X, allow_3d=True, ensure_all_finite="allow-nan")
        return self

    def transform(self, X):
        """
        Transform the data using the specified interpolation method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The input data to be transformed.

        Returns
        -------
        ndarray of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The transformed data after applying the interpolation method.
        """
        X = self._validate_data(
            X, reset=False, allow_3d=True, ensure_all_finite="allow-nan"
        )
        check_is_fitted(self)

        Method = check_option(_INTERPOLATE_METHOD, self.method, "method")
        X = check_array(
            X, allow_3d=True, allow_eos=False, ensure_all_finite="allow-nan"
        )
        index = np.arange(X.shape[-1])
        new_shape = int(np.prod(X.shape[:-1]))
        filled = np.empty_like(X).reshape(new_shape, -1)
        for i, ts in enumerate(X.reshape(new_shape, -1)):
            valid = np.isfinite(ts)
            filled[i] = Method(index[valid], ts[valid])(index)
        return filled.reshape(X.shape)
        # return interpolate(X, self.method)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class Truncate(TransformerMixin, BaseEstimator):
    """
    A transformer that truncates the input data based on the end of series indicators.
    """

    def fit(self, X, y=None):
        """
        Fit the model to the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The input data to fit the model.
        y : array-like, optional
            The target values. Ignored.

        Returns
        -------
        object
            Returns the instance of the fitted model.
        """
        self._validate_data(
            X, allow_3d=True, allow_eos=True, ensure_all_finite="allow-nan"
        )
        return self

    def transform(self, X):
        """
        Transform the input data X according to the fitted model.

        Parameters
        ----------
        X : array-like
            Input data to transform.

        Returns
        -------
        array-like
            Transformed input data.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X, reset=False, allow_3d=True, allow_eos=True, ensure_all_finite="allow-nan"
        )
        eos = np.nonzero(is_end_of_series(X))[-1]
        if eos.size > 0:
            return X[..., : np.min(eos)]
        else:
            return X

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class MinMaxScale(TransformerMixin, BaseEstimator):
    """
    Normalize time series, ensuring that each value within a specified minimum and maximum range.

    Parameters
    ----------
    min : float, optional
        The minimum value.
    max : float, optional
        The maximum value.

    Examples
    --------

    >>> from wildboar.datasets import load_gun_point
    >>> from wildboar.datasets.preprocess import MinMaxScale
    >>> X, _ = load_gun_point()
    >>> MinMaxScale().fit_transform(X).shape
    (200, 150)
    """

    _parameter_constraints = {
        "min": [Interval(numbers.Real, None, None, closed="neither")],
        "max": [Interval(numbers.Real, None, None, closed="neither")],
    }

    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the model to the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The input data to fit the model.
        y : array-like, optional
            The target values. Ignored.

        Returns
        -------
        object
            Returns the instance of the fitted model.
        """
        self._validate_data(X, allow_3d=True, ensure_all_finite="allow-nan")
        if self.min > self.max:
            raise ValueError()
        return self

    def transform(self, X):
        """
        Transform the data using the specified interpolation method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The input data to be transformed.

        Returns
        -------
        ndarray of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The transformed data after applying the interpolation method.
        """
        X = self._validate_data(
            X, reset=False, allow_3d=True, ensure_all_finite="allow-nan"
        )
        check_is_fitted(self)
        X_min = np.nanmin(X, axis=-1, keepdims=True)
        X_max = np.nanmax(X, axis=-1, keepdims=True)
        X = (X - X_min) / (X_max - X_min)
        return X * (self.max - self.min) + self.min

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class MaxAbsScale(TransformerMixin, BaseEstimator):
    """Scale each time series by its maximum absolute value."""

    def fit(self, X, y=None):
        """
        Fit the model to the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The input data to fit the model.
        y : array-like, optional
            The target values. Ignored.

        Returns
        -------
        object
            Returns the instance of the fitted model.
        """
        self._validate_data(X, allow_3d=True)
        return self

    def transform(self, X):
        """
        Transform the data using the specified interpolation method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The input data to be transformed.

        Returns
        -------
        ndarray of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The transformed data after applying the interpolation method.
        """
        X = self._validate_data(X, reset=False, allow_3d=True)
        check_is_fitted(self)
        X_max = np.nanmax(np.abs(X), axis=-1, keepdims=True)
        return X / X_max


class Standardize(TransformerMixin, BaseEstimator):
    """Standardize time series with zero mean and unit standard deviation."""

    def fit(self, X, y=None):
        """
        Fit the model to the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The input data to fit the model.
        y : array-like, optional
            The target values. Ignored.

        Returns
        -------
        object
            Returns the instance of the fitted model.
        """
        self._validate_data(X, allow_3d=True, ensure_all_finite="allow-nan")
        return self

    def transform(self, X):
        """
        Transform the data using the specified interpolation method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The input data to be transformed.

        Returns
        -------
        ndarray of shape (n_samples, n_dims, n_timestep) or (n_samples, n_timestep)
            The transformed data after applying the interpolation method.
        """
        X = self._validate_data(
            X, reset=False, allow_3d=True, ensure_all_finite="allow-nan"
        )
        check_is_fitted(self)
        std = np.nanstd(X, axis=-1, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        return (X - np.nanmean(X, axis=-1, keepdims=True)) / std

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


_PREPROCESS = {
    "standardize": standardize,
    "normalize": standardize,
    "minmax_scale": minmax_scale,
    "maxabs_scale": maxabs_scale,
    "truncate": truncate,
    "downsample": piecewice_aggregate_approximation,
    "downsample-25": partial(piecewice_aggregate_approximation, n_intervals=0.25),
    "downsample-50": partial(piecewice_aggregate_approximation, n_intervals=0.5),
    "interpolate": partial(interpolate, method="linear"),
    "interpolate-cubic": partial(interpolate, method="cubic"),
    "interpolate-pchip": partial(interpolate, method="pchip"),
}

# TODO: If we ever bump the requirement of scipy, drop this check
if __SCIPY_VERSION >= (1, 13, 0):
    _PREPROCESS["interpolate-akima"] = partial(interpolate, method="akima")
    _PREPROCESS["interpolate-makima"] = partial(interpolate, method="makima")
