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
import numpy as np


def named_preprocess(name):
    if name in _PREPROCESS:
        return _PREPROCESS[name]
    else:
        raise ValueError("preprocess (%s) does not exists" % name)


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
    return (x - np.nanmean(x, axis=-1, keepdims=True)) / np.nanstd(
        x, axis=-1, keepdims=True
    )


normalize = standardize
normalize.__doc__ = standardize.__doc__


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
    x_max = np.nanmax(np.abs(x), axis=-1, keepdims=True)
    return x / x_max


_PREPROCESS = {
    "standardize": standardize,
    "normalize": normalize,
    "minmax_scale": minmax_scale,
    "maxabs_scale": maxabs_scale,
}
