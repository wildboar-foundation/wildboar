# Authors: Isak Samsten
# License: BSD 3 clause
"""Utilities for preprocessing time series."""

import numbers
from functools import partial

import numpy as np
from sklearn.base import OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import Interval

from .. import iseos
from ..base import BaseEstimator
from ..transform._sax import piecewice_aggregate_approximation
from ..utils.validation import check_array, check_option

__all__ = [
    "standardize",
    "minmax_scale",
    "maxabs_scale",
    "truncate",
    "named_preprocess",
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
    x = check_array(x, allow_3d=True, force_all_finite="allow-nan")
    return (x - np.nanmean(x, axis=-1, keepdims=True)) / np.nanstd(
        x, axis=-1, keepdims=True
    )


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
    if min > max:
        raise ValueError("min must be smaller than max.")
    x = check_array(x, allow_3d=True, force_all_finite="allow-nan")
    x_min = np.nanmin(x, axis=-1, keepdims=True)
    x_max = np.nanmax(x, axis=-1, keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    return x * (max - min) + min


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
    x = check_array(x, allow_3d=True, force_all_finite="allow-nan")
    x_max = np.nanmax(np.abs(x), axis=-1, keepdims=True)
    return x / x_max


def truncate(x, n_shortest=None):
    """
    Truncate x to the shortest sequence.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The samples.
    n_shortest : int, optional
        The maximum size.

    Returns
    -------
    ndarray of shape (n_samples, n_shortest) or (n_samples, n_dims, n_shortest)
        The truncated samples.
    """
    x = check_array(x, allow_3d=True, allow_eos=True, force_all_finite="allow-nan")
    if n_shortest is None:
        eos = np.nonzero(iseos(x))[-1]
        if eos.size > 0:
            return x[..., : np.min(eos)]
        else:
            return x
    else:
        if n_shortest > x.shape[-1]:
            raise ValueError("n_shortest > x.shape[-1]")
        return x[..., :n_shortest]


_PREPROCESS = {
    "standardize": standardize,
    "normalize": standardize,
    "minmax_scale": minmax_scale,
    "maxabs_scale": maxabs_scale,
    "truncate": truncate,
    "downsample": piecewice_aggregate_approximation,
    "downsample-25": partial(piecewice_aggregate_approximation, n_intervals=0.25),
    "downsample-50": partial(piecewice_aggregate_approximation, n_intervals=0.5),
}


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
        x = np.sqrt(self._validate_data(x, allow_3d=False).clip(min=0))

        self.mu_ = x.mean(axis=0)
        self.sigma_ = x.std(axis=0) + (x == 0).mean() ** self.exp + 1e-8
        return self

    def transform(self, x):
        x = np.sqrt(self._validate_data(x, allow_3d=False).clip(min=0))

        x = x - self.mu_
        if self.mask_zero:
            x *= x != 0

        x /= self.sigma_

        return x
