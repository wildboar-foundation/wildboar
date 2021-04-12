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

# Authors: Isak Samsten

import math

import numpy as np

from ._dtw_distance import (
    _dtw_alignment,
    _dtw_distance,
    _dtw_envelop,
    _dtw_lb_keogh,
    _dtw_pairwise_distance,
)

__all__ = [
    "dtw_alignment",
    "dtw_distance",
    "dtw_mapping",
    "dtw_envelop",
    "dtw_pairwise_distance",
    "dtw_lb_keogh",
]


def _compute_warp_size(x_size, r, *, y_size=0):
    """Compute the warping window from array sizes and r

    Notes
    -----
    This function ensures that r is always in the range [1, max(x_size, y_size) - 1]
    """
    x_size = max(x_size, y_size)
    if isinstance(r, float):
        if not 0.0 <= r <= 1.0:
            raise ValueError("r should be in [0, 1], got %r" % r)
        return max(min(math.floor(x_size * r), x_size - 1), 1)
    elif isinstance(r, int) and 0 <= r < x_size:
        return max(1, r)
    else:
        raise ValueError("invalid warping window, got %r" % r)


def dtw_alignment(x, y, r=1.0, out=None):
    """Compute the Dynamic time warping alignment matrix

    Parameters
    ----------
    x : array-like of shape (x_timestep,)
        The first time series

    y : array-like of shape (y_timestep,)
        The second time series

    r : int or float, optional
        The warping window

        - if float in [0, 1] a fraction of max(x_timestep, y_timestep)
        - if int the exact warping window (max(1, r))

    out : array-like of shape (x_timestep, y_timestep), optional
        Store the warping path in this array.

    Returns
    -------
    alignment : ndarray of shape (x_timestep, y_timestep)
        The dynamic time warping alignment matrix

    Notes
    --------
    If only the distance between two series is required use `dtw_distance` instead

    See Also
    --------
    dtw_distance : compute the dtw distance
    """
    x = np.asarray(x)
    y = np.asarray(y)
    warp_size = _compute_warp_size(x.shape[0], r, y_size=y.shape[0])
    return _dtw_alignment(x, y, warp_size, out)


def dtw_distance(x, y, r=1.0, scale=False):
    """Compute the dynamic time warping distance

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        The first time series

    y : array-like of shape (y_timestep, )
        The second time series

    r : int or float, optional
        The warping window

        - if float in [0, 1] a fraction of max(x_timestep, y_timestep)
        - if int the exact warping window (max(1, r))

    scale : bool, optional
        If True, x and y are standardized before calculation

    Returns
    -------
    distance : float
        The dynamic time warping distance

    See Also
    --------
    dtw_alignment : compute the dtw alignment matrix
    """
    x = np.asarray(x)
    y = np.asarray(y)
    warp_size = _compute_warp_size(x.shape[0], r, y_size=y.shape[0])
    return _dtw_distance(x, y, warp_size, scale=scale)


def dtw_pairwise_distance(x, r=1.0):
    """Compute the distance between all pairs of rows

    Parameters
    ----------
    x : array-like of shape (n_samples, n_timestep)
        An array of samples

    r : float or int, optional
        The size of the warping window

        - if float in [0, 1] a fraction of x_timestep
        - if int the exact warping window (max(1, r))

    Returns
    -------
    distances : ndarray of shape (n_samples, n_samples)
        The distance between pairs of rows
    """
    x = np.asarray(x)
    if not x.ndim == 2:
        raise ValueError("requires a 2d-array")

    warp_size = _compute_warp_size(x.shape[1], r)
    return _dtw_pairwise_distance(x, warp_size)


def dtw_envelop(x, r=1.0):
    """Compute the envelop for LB_keogh

    Parameters
    ----------
    x : array-like of shape (x_timestep,)
        The time series

    r : float or int, optional
        The size of the warping window

        - if float in [0, 1] a fraction of x_timestep
        - if int the exact warping window (max(1, r))

    Returns
    -------
    lower : ndarray of shape (x_timestep,)
        The min value of the envelop

    upper : ndarray of shape (x_timestep,)
        The max value of the envelop

    References
    ----------
    Keogh, E. (2002).
        Exact indexing of dynamic time warping.
        In 28th International Conference on Very Large Data Bases.
    """
    x = np.asarray(x)
    warp_size = _compute_warp_size(x.shape[0], r)
    return _dtw_envelop(x, warp_size)


def dtw_lb_keogh(x, y=None, *, lower=None, upper=None, r=1.0):
    """The LB_keogh lower bound

    Parameters
    ----------
    x : array-like of shape (x_timestep,)
        The first time series

    y : array-like of shape (x_timestep,), optional
        The second time series (same size as x)

    lower : ndarray of shape (x_timestep,), optional
        The min value of the envelop

    upper : ndarray of shape (x_timestep,), optional
        The max value of the envelop

    r : float or int, optional
        The size of the warping window

        - if float in [0, 1] a fraction of x_timestep
        - if int the exact warping window (max(1, r))

    Returns
    -------
    min_dist : float
        The cumulative minimum distance.

    lb_keogh : ndarray of shape (x_timestep,),
        The lower bound at each time step

    Notes
    -----
    - if y=None, both lower and upper must be given
    - if y is given, lower and upper are ignored
    - if lower and upper is given and y=None, r is ignored

    References
    ----------
    Keogh, E. (2002).
        Exact indexing of dynamic time warping.
        In 28th International Conference on Very Large Data Bases.
    """
    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)
        if y.shape[0] != x.shape[0]:
            raise ValueError("invalid shape, got (%d, %d)" % (x.shape[0], y.shape[0]))
        lower, upper = dtw_envelop(y, r)
    elif lower is None or upper is None:
        raise ValueError("both y, lower and upper can't be None")

    if lower.shape[0] != upper.shape[0] or lower.shape[0] != x.shape[0]:
        raise ValueError(
            "invalid shape for lower (%d), upper (%d) and x (%d)"
            % (lower.shape[0], upper.shape[0], x.shape[0])
        )
    warp_size = _compute_warp_size(x.shape[0], r)
    return _dtw_lb_keogh(x, lower, upper, warp_size)


def dtw_mapping(x=None, y=None, *, alignment=None, r=1, return_index=False):
    """Compute the optimal warping path between two series or from a given
    alignment matrix

    Parameters
    ----------
    x : array-like of shape (x_timestep,), optional
        The first time series

    y : array-like of shape (y_timestep,), optional
        The second time series

    alignment : ndarray of shape (x_timestep, y_timestep), optional
        Precomputed alignment

    r : float or int, optional
        - if float, the warping path is a fraction of

    return_index : bool, optional
        Return the indices of the warping path

    Returns
    -------
    indicator : ndarray of shape (x_timestep, y_timestep)
        Boolean array with the dtw path

    (x_indices, y_indices) : tuple, optional
        The indices of the first and second dimension of
        the optimal alignment path.

    Notes
    -----
    - either x and y or alignment must be provided
    - if alignment is given x and y are ignored
    - if alignment is given r is ignored
    """
    if alignment is None:
        if x is None or y is None:
            raise ValueError("if alignment=None, neither x or y can be None")
        alignment = dtw_alignment(x, y, r=r)

    indicator = np.zeros(alignment.shape).astype(bool)
    i = alignment.shape[0] - 1
    j = alignment.shape[1] - 1
    while i > 0 or j > 0:
        indicator[i, j] = True
        option_diag = alignment[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = alignment[i - 1, j] if i > 0 else np.inf
        option_left = alignment[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    indicator[0, 0] = True
    if return_index:
        return indicator, indicator.nonzero()
    else:
        return indicator
