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
from sklearn.utils import deprecated

from ..utils.validation import check_array
from . import pairwise_distance
from ._dtw_distance import _dtw_alignment, _dtw_envelop, _dtw_lb_keogh

__all__ = [
    "dtw_alignment",
    "wdtw_alignment",
    "dtw_distance",
    "wdtw_distance",
    "ddtw_distance",
    "wddtw_distance",
    "dtw_mapping",
    "dtw_envelop",
    "dtw_pairwise_distance",
    "dtw_lb_keogh",
    "jeong_weight",
]


def _compute_warp_size(x_size, r, *, y_size=0):
    x_size = max(x_size, y_size)
    if not 0.0 <= r <= 1.0:
        raise ValueError("r should be in [0, 1], got %r" % r)
    return max(math.floor(x_size * r), 1)


def dtw_distance(x, y, *, r=1.0):
    """Compute the dynamic time warping distance

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        The first time series

    y : array-like of shape (y_timestep, )
        The second time series

    r : float, optional
        The warping window in [0, 1] as a fraction of max(x_timestep, y_timestep)

    Returns
    -------
    distance : float
        The dynamic time warping distance

    See Also
    --------
    dtw_alignment : compute the dtw alignment matrix
    """
    return pairwise_distance(x, y, metric="dtw", metric_params={"r": r})


def ddtw_distance(x, y, *, r=1.0):
    """Compute the derivative dynamic time warping distance

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        The first time series

    y : array-like of shape (y_timestep, )
        The second time series

    r : float, optional
        The warping window in [0, 1] as a fraction of max(x_timestep, y_timestep)

    Returns
    -------
    distance : float
        The dynamic time warping distance

    See Also
    --------
    dtw_distance : compute the dtw distance
    """
    return pairwise_distance(x, y, metric="ddtw", metric_params={"r": r})


def wdtw_distance(x, y, *, r=1.0, g=0.05):
    """Compute the weighted dynamic time warping distance

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        The first time series

    y : array-like of shape (y_timestep, )
        The second time series

    r : float, optional
        The warping window in [0, 1] as a fraction of ``max(x_timestep, y_timestep)``

    g : float, optional
        Penalization for points deviating the diagonal.

    Returns
    -------
    distance : float
        The dynamic time warping distance

    See Also
    --------
    dtw_distance : compute the dtw distance
    """
    return pairwise_distance(x, y, metric="wdtw", metric_params={"r": r, "g": g})


def wddtw_distance(x, y, *, r=1.0, g=0.05):
    """Compute the weighted derivative dynamic time warping distance

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        The first time series

    y : array-like of shape (y_timestep, )
        The second time series

    r : float, optional
        The warping window in [0, 1] as a fraction of ``max(x_timestep, y_timestep)``

    g : float, optional
        Penalization for points deviating the diagonal.

    Returns
    -------
    distance : float
        The dynamic time warping distance

    See Also
    --------
    dtw_distance : compute the dtw distance
    """
    return pairwise_distance(x, y, metric="wddtw", metric_params={"r": r, "g": g})


@deprecated(
    "Function 'dtw_pairwise_distance' was deprectad in 1.1 and will be removed in 1.2."
    "Use 'pairwise_distance(x, metric=\"dtw\")' instead."
)
def dtw_pairwise_distance(x, r=1.0):
    """Compute the distance between all pairs of rows

    Parameters
    ----------
    x : array-like of shape (n_samples, n_timestep)
        An array of samples

    r : float or int, optional
        The warping window in [0, 1] as a fraction of max(x_timestep, y_timestep)

    Returns
    -------
    distances : ndarray of shape (n_samples, n_samples)
        The distance between pairs of rows
    """
    return pairwise_distance(x, metric="dtw", metric_params={"r": r})


def dtw_envelop(x, *, r=1.0):
    """Compute the envelop for LB_keogh

    Parameters
    ----------
    x : array-like of shape (x_timestep,)
        The time series

    r : float, optional
        The warping window in [0, 1] as a fraction of max(x_timestep, y_timestep)

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
    x = check_array(x, ensure_1d=True, dtype=float, contiguous=True)
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

    r : float, optional
        The warping window in [0, 1] as a fraction of max(x_timestep, y_timestep)

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
    x = check_array(x, ensure_1d=True, contiguous=True, dtype=float)
    if x.ndim != 1:
        raise ValueError("invalid ndim, got (%d)" % x.ndim)
    if y is not None:
        y = check_array(y, ensure_1d=True, contiguous=True, dtype=float)
        if y.shape[0] != x.shape[0]:
            raise ValueError("invalid shape, got (%d, %d)" % (x.shape[0], y.shape[0]))
        lower, upper = dtw_envelop(y, r=r)
    elif lower is None or upper is None:
        raise ValueError("both y, lower and upper can't be None")

    lower = check_array(lower, ensure_1d=True, contiguous=True, dtype=float)
    upper = check_array(upper, ensure_1d=True, contiguous=True, dtype=float)
    if lower.shape[0] != upper.shape[0] or lower.shape[0] != x.shape[0]:
        raise ValueError(
            "invalid shape for lower (%d), upper (%d) and x (%d)"
            % (lower.shape[0], upper.shape[0], x.shape[0])
        )
    warp_size = _compute_warp_size(x.shape[0], r)
    return _dtw_lb_keogh(x, lower, upper, warp_size)


def dtw_alignment(x, y, *, r=1.0, weight=None, out=None):
    """Compute the Dynamic time warping alignment matrix

    Parameters
    ----------
    x : array-like of shape (x_timestep,)
        The first time series

    y : array-like of shape (y_timestep,)
        The second time series

    r : float, optional
        The warping window in [0, 1] as a fraction of max(x_timestep, y_timestep)

    out : array-like of shape (x_timestep, y_timestep), optional
        Store the warping path in this array.

    weight : array-like of shape (max(x_timestep, y_timestep), ), optional
        A weighting vector to penalize warping.

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

    References
    ----------
    Jeong, Y., Jeong, M., Omitaomu, O. (2021)
        Weighted dynamic time warping for time series classification.
        Pattern Recognition 44, 2231-2240
    """
    x = check_array(x, ensure_2d=False, dtype=float)
    y = check_array(y, ensure_2d=False, dtype=float)
    warp_size = _compute_warp_size(x.shape[0], r, y_size=y.shape[0])
    if weight is not None:
        weight = np.asarray(weight, dtype=float)
        if weight.shape[0] != max(x.shape[0], y.shape[0]):
            raise ValueError("invalid weight array.")

    return _dtw_alignment(x, y, warp_size, weight, out)


def wdtw_alignment(x, y, *, r=1.0, g=0.5, out=None):
    """Weighted dynamic time warping alignment

    Parameters
    ----------
    x : array-like of shape (x_timestep,)
        The first time series

    y : array-like of shape (y_timestep,)
        The second time series

    r : float, optional
        The warping window in [0, 1] as a fraction of max(x_timestep, y_timestep)

    g : float, optional
        Weighting described by Jeong et. al. (2011) using :math:`g` as penalty control.

        .. math:: w(x)=\\frac{w_{max}}{1+e^{-g(x-m/2)}},

    out : array-like of shape (x_timestep, y_timestep), optional
        Store the warping path in this array.

    Returns
    -------
    alignment : ndarray of shape (x_timestep, y_timestep)
        The dynamic time warping alignment matrix

    References
    ----------
    Jeong, Y., Jeong, M., Omitaomu, O. (2021)
        Weighted dynamic time warping for time series classification.
        Pattern Recognition 44, 2231-2240
    """
    weight = jeong_weight(max(x.shape[0], y.shape[0]), g)
    return dtw_alignment(x, y, r=r, weight=weight, out=out)


def jeong_weight(n, g=0.05):
    """Weighting described by Jeong et. al. (2011) using g as the penalty control.

    .. math:: w(x)=\\frac{1}{1+e^{-g(x-m/2)}}

    Parameters
    ----------
    n : int
        The number of weights.

    g : float, optional
        Penalty control.

    Returns
    -------
    weight : ndarray of shape (n, )
        The weights

    References
    ----------
    Jeong, Y., Jeong, M., Omitaomu, O. (2021)
        Weighted dynamic time warping for time series classification.
        Pattern Recognition 44, 2231-2240
    """
    weight = 1.0 / (1.0 + np.exp(-g * (np.arange(n, dtype=float) - n / 2.0)))
    return weight


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

    r : float, optional
        The warping window in [0, 1] as a fraction of max(x_timestep, y_timestep)

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
