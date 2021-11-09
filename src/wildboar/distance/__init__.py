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

from wildboar.utils import check_array
from wildboar.utils.decorators import array_or_scalar

from . import _distance, _dtw_distance, _euclidean_distance

__all__ = [
    "distance",
    "matches",
    "pairwise_subsequence_distance",
    "paired_subsequence_distance",
    "pairwise_subsequence_match",
    "paired_subsequence_match",
    "pairwise_distance",
    "paired_distance",
]

_SUBSEQUENCE_DISTANCE_MEASURE = {
    "euclidean": _euclidean_distance.EuclideanSubsequenceDistanceMeasure,
    "scaled_euclidean": _euclidean_distance.ScaledEuclideanSubsequenceDistanceMeasure,
    "dtw": _dtw_distance.DtwSubsequenceDistanceMeasure,
    "scaled_dtw": _dtw_distance.ScaledDtwSubsequenceDistanceMeasure,
}

_DISTANCE_MEASURE = {
    "euclidean": _euclidean_distance.EuclideanDistanceMeasure,
    "dtw": _dtw_distance.DtwDistanceMeasure,
}


def _validate_subsequence(y):
    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single timestep or array.reshape(1, -1) "
                "if it contains a single sample.".format(y)
            )
        elif y.ndim == 2:
            y = list(y.astype(np.double))
        else:
            raise ValueError(
                "Expected 2D array, got {}D array instead:\narray={}.\n".format(
                    y.ndim, y
                )
            )
    else:
        if all(isinstance(e, (int, float)) for e in y):
            y = [np.array(y, dtype=np.double)]
        else:
            y = [np.array(e, dtype=np.double) for e in y]

    return y


@array_or_scalar(squeeze=True)
def pairwise_subsequence_distance(
    y,
    x,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    return_index=False,
    n_jobs=None,
):
    """Compute the minimum sliding distance between shapelets and time series

    Parameters
    ----------
    y : list or ndarray of shape (n_samples, n_timestep)
        Input time series.

        - if list, a list of array-like of shape (n_timestep, )

    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The input data

    dim : int, optional
        The dim to search for shapelets

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
        [description], by default None
    """
    y = _validate_subsequence(y)
    x = check_array(x, allow_multivariate=True, dtype=np.double)
    distance_measure = _SUBSEQUENCE_DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError()

    metric_params = metric_params or {}
    min_dist, min_ind = _distance._pairwise_subsequence_distance(
        y,
        x,
        dim,
        distance_measure(**metric_params),
        n_jobs,
    )
    if return_index:
        return min_dist, min_ind
    else:
        return min_dist


def paired_subsequence_distance(
    y,
    x,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    n_jobs=None,
):
    raise NotImplementedError()


def subsequence_match(
    y,
    x,
    threshold,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    return_distance=False,
    n_jobs=None,
):
    y = _validate_subsequence(y)
    if len(y) > 1:
        raise ValueError("....")
    y = y[0]
    x = check_array(x, allow_multivariate=True, dtype=np.double)
    distance_measure = _SUBSEQUENCE_DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError()
    metric_params = metric_params if metric_params is not None else {}
    indicies, distances = _distance._subsequence_match(
        y,
        x,
        threshold,
        dim,
        distance_measure(**metric_params),
        n_jobs,
    )
    if return_distance:
        return indicies, distances
    else:
        return indicies


def paired_subsequence_match(
    y,
    x,
    threshold,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    return_distance=False,
    n_jobs=None,
):
    y = _validate_subsequence(y)
    x = check_array(x, allow_multivariate=True, dtype=np.double)
    if len(y) != x.shape[0]:
        raise ValueError("n_samples")

    distance_measure = _SUBSEQUENCE_DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError()
    metric_params = metric_params if metric_params is not None else {}
    indicies, distances = _distance._paired_subsequence_match(
        y,
        x,
        threshold,
        dim,
        distance_measure(**metric_params),
        n_jobs,
    )
    if return_distance:
        return indicies, distances
    else:
        return indicies


def paired_distance(
    x,
    y,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    n_jobs=None,
):
    x = check_array(x, allow_multivariate=True, dtype=np.double)
    y = check_array(y, allow_multivariate=True, dtype=np.double)
    if x.ndim != y.ndim:
        raise ValueError("dim")
    if x.shape[0] != y.shape[0]:
        raise ValueError("n_samples")

    distance_measure = _DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError()

    metric_params = metric_params or {}
    distance_measure = distance_measure(**metric_params)
    if x.shape[x.ndim - 1] != x.shape[x.ndim - 1] and not distance_measure.is_elastic:
        raise ValueError()

    return _distance._paired_distance(
        x,
        y,
        dim,
        distance_measure,
        n_jobs,
    )


def pairwise_distance(
    x,
    y,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    n_jobs=None,
):
    distance_measure = _DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError()

    metric_params = metric_params or {}
    distance_measure = distance_measure(**metric_params)
    if x is y:
        x = check_array(x, allow_multivariate=True, dtype=np.double)
        if not 0 >= dim < x.ndim:
            raise ValueError()
        return _distance._singleton_pairwise_distance(x, dim, distance_measure, n_jobs)
    else:
        x = check_array(x, allow_multivariate=True, dtype=np.double)
        y = check_array(y, allow_multivariate=True, dtype=np.double)
        if x.ndim != y.ndim:
            raise ValueError("dim")

        if not 0 >= dim < x.ndim:
            raise ValueError()

        if (
            x.shape[x.ndim - 1] != y.shape[y.ndim - 1]
            and not distance_measure.is_elastic
        ):
            raise ValueError("timestep")

        return _distance._pairwise_distance(
            x,
            y,
            dim,
            distance_measure,
            n_jobs,
        )


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
