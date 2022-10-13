# Authors: Isak Samsten
# License: BSD 3 clause

from functools import partial

import numpy as np

from .. import iseos
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
    return check_option(_PREPROCESS, name, "name")


def standardize(x):
    """Scale x along the time dimension to have zero mean and unit standard deviation

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The dataset

    Returns
    -------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The standardized dataset
    """
    x = check_array(x, allow_3d=True, force_all_finite="allow-nan")
    return (x - np.nanmean(x, axis=-1, keepdims=True)) / np.nanstd(
        x, axis=-1, keepdims=True
    )


def minmax_scale(x, min=0, max=1):
    """Scale x along the time dimension so that each value is between min and max

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The dataset

    min : float, optional
        The minimum value

    max : float, optional
        The maximum value

    Returns
    -------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The transformed dataset
    """
    if min > max:
        raise ValueError("min must be smaller than max.")
    x = check_array(x, allow_3d=True, force_all_finite="allow-nan")
    x_min = np.nanmin(x, axis=-1, keepdims=True)
    x_max = np.nanmax(x, axis=-1, keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    return x * (max - min) + min


def maxabs_scale(x):
    """Scale each time series by its maximum absolute value.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The dataset

    Returns
    -------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The transformed dataset
    """
    x = check_array(x, allow_3d=True, force_all_finite="allow-nan")
    x_max = np.nanmax(np.abs(x), axis=-1, keepdims=True)
    return x / x_max


def truncate(x, n_shortest=None):
    """Truncate x to the shortest sequence.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The dataset

    n_shortest : int, optional
        The maximum size

    Returns
    -------
    x : ndarray of shape (n_samples, n_shortest) or (n_samples, n_dims, n_shortest)
        The truncated dataset
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
