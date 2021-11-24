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
import warnings

import numpy as np
from sklearn.utils.deprecation import deprecated

from wildboar.utils import check_array
from wildboar.utils.decorators import array_or_scalar

from . import _distance, _dtw_distance, _euclidean_distance, _mass

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
    "mass": _mass.ScaledMassSubsequenceDistanceMeasure,
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
    """Compute the minimum subsequence distance between subsequences and time series

    Parameters
    ----------
    y : list or ndarray of shape (n_subsequences, n_timestep)
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
        An array of shape (n_subsequences, n_samples) with the minumum distance between
        each subsequence and each sample.

    indices : int, ndarray
        An array of shape (n_subsequences, n_samples) with the start position of the
        best match between each subsequence and time series
    """
    y = _validate_subsequence(y)
    x = check_array(x, allow_multivariate=True, dtype=np.double)
    for s in y:
        if s.shape[0] > x.shape[-1]:
            raise ValueError(
                "invalid subsequnce shape (%d > %d)" % (s.shape[0], x.shape[-1])
            )

    distance_measure = _SUBSEQUENCE_DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError("unsupported metric (%r)" % metric)

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
    return_index=False,
    n_jobs=None,
):
    """Compute the minimum subsequence distance between the i:th subsequence and time
    series

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

    n_jobs : int, optional
        The number of parallel jobs to run. Ignored

    Returns
    -------
    dist : float, ndarray
        An array of shape (n_samples, ) with the minumum distance between the i:th
        subsequence and the i:th sample

    indices : int, ndarray
        An array of shape (n_samples, ) with the index of the best matching position
        of the i:th subsequence and the i:th sample
    """
    y = _validate_subsequence(y)
    x = check_array(x, allow_multivariate=True, dtype=np.double)
    for s in y:
        if s.shape[0] > x.shape[-1]:
            raise ValueError(
                "invalid subsequnce shape (%d > %d)" % (s.shape[0], x.shape[-1])
            )
    distance_measure = _SUBSEQUENCE_DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError("unsupported metric (%r)" % metric)

    if n_jobs is not None:
        warnings.warn("n_jobs is not yet supported.", UserWarning)

    metric_params = metric_params or {}
    min_dist, min_ind = _distance._paired_subsequence_distance(
        y, x, dim, distance_measure(**metric_params)
    )
    if return_index:
        return min_dist, min_ind
    else:
        return min_dist


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
    """Find the positions where the distance is less than the threshold between the
    subsequence and all time series.

    Parameters
    ----------
    y : array-like of shape (n_timestep, )
        The subsequence

    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The input data

    threshold : float
        The distance threshold used to consider a subsequence matching

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

    return_distance : bool, optional
        - if True, return the distance of the match

    n_jobs : int, optional
        The number of parallel jobs to run. Ignored

    Returns
    -------
    indicies : list
        A list of shape (n_samples, ) of ndarray with start index of matching
        subsequences

    distance : list
        A list of shape (n_samples, ) of ndarray with distance at the matching position.
    """
    y = _validate_subsequence(y)
    if len(y) > 1:
        raise ValueError("a single sample expected")
    y = y[0]
    x = check_array(x, allow_multivariate=True, dtype=np.double)
    if y.shape[0] > x.shape[-1]:
        raise ValueError(
            "invalid subsequnce shape (%d > %d)" % (y.shape[0], x.shape[-1])
        )
    distance_measure = _SUBSEQUENCE_DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError("unsupported metric (%r)" % metric)
    metric_params = metric_params if metric_params is not None else {}

    if n_jobs is not None:
        warnings.warn("n_jobs is not yet supported.", UserWarning)

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
    """Compute the minimum subsequence distance between the i:th subsequence and time
    series

    Parameters
    ----------
    y : list or ndarray of shape (n_samples, n_timestep)
        Input time series.

        - if list, a list of array-like of shape (n_timestep, )

    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The input data

    threshold : float
        The distance threshold used to consider a subsequence matching

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

    return_distance : bool, optional
        - if True, return the distance of the match

    n_jobs : int, optional
        The number of parallel jobs to run. Ignored

    Returns
    -------
    indicies : list
        A list of shape (n_samples, ) of ndarray with start index of matching
        subsequences

    distance : list
        A list of shape (n_samples, ) of ndarray with distance at the matching position.
    """
    y = _validate_subsequence(y)
    x = check_array(x, allow_multivariate=True, dtype=np.double)
    if len(y) != x.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    for s in y:
        if s.shape[0] > x.shape[-1]:
            raise ValueError(
                "invalid subsequnce shape (%d > %d)" % (s.shape[0], x.shape[-1])
            )

    if n_jobs is not None:
        warnings.warn("n_jobs is not yet supported.", UserWarning)

    distance_measure = _SUBSEQUENCE_DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError("unsupported metric (%r)" % metric)
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
    """Compute the distance between the i:th time series

    Parameters
    ----------
    y : : ndarray of shape (n_samples, n_timestep) or (y_samples, n_dims, n_timestep)
        The input data

    x : ndarray of shape (n_samples, n_timestep) or (x_samples, n_dims, n_timestep)
        The input data

    dim : int, optional
        The dim to compute distance

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

    n_jobs : int, optional
        The number of parallel jobs.

    Returns
    -------
    dist : float or ndarray
        An array of shape (y_samples, )
    """
    x = check_array(x, allow_multivariate=True, dtype=np.double)
    y = check_array(y, allow_multivariate=True, dtype=np.double)
    if x.ndim != y.ndim:
        raise ValueError(
            "x (%dD-array) and y (%dD-array) are not compatible" % (x.ndim, y.ndim)
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    if n_jobs is not None:
        warnings.warn("n_jobs is not yet supported.", UserWarning)

    distance_measure = _DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError("unsupported metric (%r)" % metric)

    metric_params = metric_params or {}
    distance_measure = distance_measure(**metric_params)
    if x.shape[x.ndim - 1] != x.shape[x.ndim - 1] and not distance_measure.is_elastic:
        raise ValueError(
            "illegal n_timestep (%r != %r) for non-elastic distance measure"
            % (x.shape[x.ndim - 1], y.shape[y.ndim - 1])
        )

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
    """Compute the distance between subsequences and time series

    Parameters
    ----------
    y : : ndarray of shape (y_samples, n_timestep) or (y_samples, n_dims, n_timestep)
        The input data

    x : ndarray of shape (x_samples, n_timestep) or (x_samples, n_dims, n_timestep)
        The input data

    dim : int, optional
        The dim to compute distance

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

    n_jobs : int, optional
        The number of parallel jobs.

    Returns
    -------
    dist : float or ndarray
        An array of shape (y_samples, x_samples)
    """
    distance_measure = _DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError("unsupported metric (%r)" % metric)

    metric_params = metric_params or {}
    distance_measure = distance_measure(**metric_params)
    if x is y:
        x = check_array(x, allow_multivariate=True, dtype=np.double)
        if not 0 >= dim < x.ndim:
            raise ValueError("illegal dim (0>=%d<%d)" % (dim, x.ndim))
        return _distance._singleton_pairwise_distance(x, dim, distance_measure, n_jobs)
    else:
        x = check_array(x, allow_multivariate=True, dtype=np.double)
        y = check_array(y, allow_multivariate=True, dtype=np.double)
        if x.ndim != y.ndim:
            raise ValueError(
                "x (%dD-array) and y (%dD-array) are not compatible" % (x.ndim, y.ndim)
            )

        if not 0 >= dim < x.ndim:
            raise ValueError("illegal dim (0>=%d<%d)" % (dim, x.ndim))

        if (
            x.shape[x.ndim - 1] != y.shape[y.ndim - 1]
            and not distance_measure.is_elastic
        ):
            raise ValueError(
                "illegal n_timestep (%r != %r) for non-elastic distance measure"
                % (x.shape[x.ndim - 1], y.shape[y.ndim - 1])
            )

        return _distance._pairwise_distance(
            x,
            y,
            dim,
            distance_measure,
            n_jobs,
        )


@deprecated(extra="Will be removed in 1.2")
def distance(
    y,
    x,
    *,
    dim=0,
    sample=None,
    metric="euclidean",
    metric_params=None,
    subsequence_distance=True,
    return_index=False,
):
    """Computes the distance between y and the samples of x

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
    x = (x[sample] if x.ndim > 1 else x.reshape(1, -1),)
    if subsequence_distance:
        return pairwise_subsequence_distance(
            y.reshape(1, -1),
            x,
            dim=dim,
            metric=metric,
            metric_params=metric_params,
            return_index=return_index,
            n_jobs=None,
        )
    else:
        return pairwise_distance(
            y.reshape(1, -1),
            x,
            dim=dim,
            metric=metric,
            metric_params=metric_params,
            n_jobs=None,
        )


@deprecated(extra="Will be removed in 1.2")
def matches(
    y,
    x,
    threshold,
    *,
    dim=0,
    sample=None,
    metric="euclidean",
    metric_params=None,
    return_distance=False,
):
    """Find matches

    Parameters
    ----------
    y : array-like of shape (x_timestep, )
        A 1-dimensional float array

    x : array-like of shape (n_samples, n_timesteps) or (n_samples, n_dims, n_timesteps)
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
    return subsequence_match(
        y,
        x[sample] if x.ndim > 1 else x.reshape(1, -1),
        threshold,
        dim=dim,
        metric=metric,
        metric_params=metric_params,
        return_distance=return_distance,
        n_jobs=None,
    )
