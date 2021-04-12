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

from . import _distance, dtw

__all__ = ["distance", "matches", "dtw"]


def paired_distance(
    x,
    y,
    *,
    metric="euclidean",
    metric_params=None,
    subsequence_distance=True,
):
    raise NotImplementedError()


def distance(
    x,
    y,
    *,
    dim=0,
    sample=None,
    metric="euclidean",
    metric_params=None,
    subsequence_distance=True,
    return_index=False,
):
    """Computes the distance between x and the samples of y

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        A 1-dimensional float array

    y : array-like of shape (n_samples, n_timesteps) or (n_samples, n_dims, n_timesteps)

    dim : int, optional
        The time series dimension to search

    sample : int or array-like, optional
        The samples to compare to

        - if ``sample=None`` the distances to all samples in data is returned
        - if sample is an int the distance to that sample is returned
        - if sample is an array-like the distance to all samples in sample are returned
        - if ``n_samples=1``, ``samples`` is an int or ``len(samples)==1`` a scalar is
          returned
        - otherwise an array is returned

    metric : {'euclidean', 'scaled_euclidean', 'dtw', 'scaled_dtw'} or callable, optional # noqa: E501
        The distance metric

        - if str use optimized implementations of the named distance measure
        - if callable a function taking two arrays as input

    metric_params: dict, optional
        Parameters to the metric

        - 'euclidean' and 'scaled_euclidean' take no parameters
        - 'dtw' and 'scaled_dtw' take a single paramater 'r'. If 'r' <= 1 it
          is interpreted as a fraction of the time series length. If > 1 it
          is interpreted as an exact time warping window. Use 'r' == 0 for
          a widow size of exactly 1.

    subsequence_distance: bool, optional
        - if True, compute the minimum subsequence distance
        - if False, compute the distance between two arrays of the same length
          unless the specified metric support unaligned arrays

    return_index : bool, optional
        - if True return the index of the best match. If there are many equally good
          matches, the first match is returned.

    Returns
    -------
    dist : float, ndarray
        The smallest distance to each time series

    indices : int, ndarray
        The start position of the best match in each time series

    See Also
    --------
    matches : find shapelets within a threshold

    Examples
    --------

    >>> from wildboar.datasets import load_two_lead_ecg
    >>> x, y = load_two_lead_ecg()
    >>> _, i = distance(x[0, 10:20], x, sample=[0, 1, 2, 3, 5, 10],
    ...                 metric="scaled_euclidean", return_index=True)
    >>> i
    [10 29  9 72 20 30]
    """
    return _distance.distance(
        x,
        y,
        dim=dim,
        sample=sample,
        metric=metric,
        metric_params=metric_params,
        subsequence_distance=subsequence_distance,
        return_index=return_index,
    )


def matches(
    x,
    y,
    threshold,
    *,
    dim=0,
    sample=None,
    metric="euclidean",
    metric_params=None,
    return_distance=False,
):
    """Return the positions in `x` (one list per `sample`) where `x` is closer than
    `threshold`.

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        A 1-dimensional float array

    y : array-like of shape (n_samples, n_timesteps) or (n_samples, n_dims, n_timesteps)
        The collection of samples

    threshold : float
        The maximum threshold to consider a match

    dim : int, optional
        The time series dimension to search

    sample : int or array-like, optional
        The samples to compare to

        - if ``sample=None`` the distances to all samples in data is returned
        - if sample is an int the distance to that sample is returned
        - if sample is an array-like the distance to all samples in sample are returned
        - if ``n_samples=1``, ``samples`` is an int or ``len(samples)==1`` a scalar is
          returned
        - otherwise an array is returned

    metric : {'euclidean', 'scaled_euclidean'}, optional
        The distance metric

    metric_params: dict, optional
        Parameters to the metric

    return_distance : bool, optional
        - if `true` return the distance of the best match.

    Returns
    -------
    dist : list
        The distances of the matching positions

    matches : list
        The start position of the matches in each time series

    Warnings
    --------
    'scaled_dtw' is not supported.
    """
    return _distance.matches(
        x, y, threshold, dim, sample, metric, metric_params, return_distance
    )
