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
import math
import numbers
import warnings

import numpy as np
from sklearn.utils.deprecation import deprecated

from wildboar.utils import check_array
from wildboar.utils.decorators import array_or_scalar, singleton

from . import _distance, _dtw_distance, _euclidean_distance, _mass, _matrix_profile

__all__ = [
    "distance",
    "matches",
    "pairwise_subsequence_distance",
    "paired_subsequence_distance",
    "subsequence_match",
    "paired_subsequence_match",
    "pairwise_distance",
    "paired_distance",
    "matrix_profile",
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

_THRESHOLD = {
    "best": lambda x: max(np.mean(x) - 2.0 * np.std(x), np.min(x)),
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
        if all(isinstance(e, (int, numbers.Real)) for e in y):
            y = [np.array(y, dtype=np.double)]
        else:
            y = [np.array(e, dtype=np.double) for e in y]

    return y


def _any_in_exclude(lst, i, exclude):
    for e in lst:
        if not (e <= i - exclude or e >= i + exclude):
            return True
    return False


def _exclude_trivial_matches(indicies, distances, exclude):
    indicies_tmp = []
    distances_tmp = []
    for index, distance in zip(indicies, distances):
        if index is None:
            indicies_tmp.append(None)
            distances_tmp.append(None)
        else:
            # For each index if index has neighbors do not include those
            sort = np.argsort(distance)
            idx = np.zeros(sort.size, dtype=bool)
            excluded = []
            for i in range(index.size):
                if not _any_in_exclude(excluded, index[sort[i]], exclude):
                    excluded.append(index[sort[i]])
                    idx[sort[i]] = True

            idx = np.array(idx)
            indicies_tmp.append(index[idx])
            distances_tmp.append(distance[idx])

    return indicies_tmp, distances_tmp


def _filter_by_max_matches(indicies, distances, max_matches):
    indicies_tmp = []
    distances_tmp = []
    for index, distance in zip(indicies, distances):
        if index is None:
            indicies_tmp.append(None)
            distances_tmp.append(None)
        else:
            idx = np.argsort(distance)[:max_matches]
            indicies_tmp.append(index[idx])
            distances_tmp.append(distance[idx])

    return indicies_tmp, distances_tmp


def _filter_by_max_dist(indicies, distances, max_dist):
    indicies_tmp = []
    distances_tmp = []
    for index, distance in zip(indicies, distances):
        if index is None:
            indicies_tmp.append(None)
            distances_tmp.append(None)
        else:
            idx = max_dist(distance)
            indicies_tmp.append(index[idx])
            distances_tmp.append(distance[idx])

    return indicies_tmp, distances_tmp


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


@array_or_scalar(squeeze=True)
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

     metric : str or callable, optional
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


@singleton
def subsequence_match(
    y,
    x,
    threshold=None,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    max_matches=None,
    exclude=None,
    return_distance=False,
    n_jobs=None,
):
    """Find the positions where the distance is less than the threshold between the
    subsequence and all time series.

    - If a `threshold` is given, the default behaviour is to return all matching
      indices in the order of occurrence
    - If no `threshold` is given, the default behaviour is to return the top 10
      matching indicies ordered by distance
    - If both `threshold` and `max_matches` are given, the top matches are returned
      ordered by distance.

    Parameters
    ----------
    y : array-like of shape (yn_timestep, )
        The subsequence

    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The input data

    threshold : str, float or callable, optional
        The distance threshold used to consider a subsequence matching. If no threshold
        is selected, `max_matches´ default to 10.

        - if float, return all matches closer than threshold
        - if callable, return all matches closer than the treshold computed by the
          threshold function, given all distances to the subsequence
        - if str, return all matches according to the named threshold.

    dim : int, optional
        The dim to search for shapelets

    metric : str or callable, optional
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

    max_matches : int, optional
        Return the top `max_matches` matches below `threshold`.

    exclude : float or int, optional
        Exclude trivial matches in the vicinity of the match.

        - if float, the exclusion zone is computed as ``math.ceil(exclude * y.size)``
        - if int, the exclusion zone is exact

        A match is considered trivial if a match with lower distance is within `exclude`
        timesteps of another match with higher distance.

    return_distance : bool, optional
        - if True, return the distance of the match

    n_jobs : int, optional
        The number of parallel jobs to run. Ignored

    Returns
    -------
    indicies : list or ndarray
        A list of shape (n_samples, ) of ndarray with start index of matching
        subsequences. If only one subsequence is given, the return value is a single
        ndarray.

    distance : list or ndarray
        A list of shape (n_samples, ) of ndarray with distance at the matching position.
        If only one subsequence is given, the return value is a single ndarray.
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

    if threshold is None:
        threshold = np.inf
        if max_matches is None:
            max_matches = 10

    if callable(threshold):
        threshold_fn = threshold

        def max_dist(d):
            return d <= threshold_fn(d)

        threshold = np.inf
    elif isinstance(threshold, str):
        threshold_fn = _THRESHOLD.get(threshold, None)
        if threshold_fn is None:
            raise ValueError("invalid threshold (%r)" % threshold)

        def max_dist(d):
            return d <= threshold_fn(d)

        threshold = np.inf
    elif not isinstance(threshold, numbers.Real):
        raise ValueError("invalid threshold (%r)" % threshold)
    else:
        max_dist = None

    if isinstance(exclude, numbers.Integral):
        if exclude < 0:
            raise ValueError("invalid exclusion (%d < 0)" % exclude)
    elif isinstance(exclude, numbers.Real):
        exclude = math.ceil(y.size * exclude)
    elif exclude is not None:
        raise ValueError("invalid exclusion (%r)" % exclude)

    indicies, distances = _distance._subsequence_match(
        y,
        x,
        threshold,
        dim,
        distance_measure(**metric_params),
        n_jobs,
    )

    if max_dist is not None:
        indicies, distances = _filter_by_max_dist(indicies, distances, max_dist)

    if exclude:
        indicies, distances = _exclude_trivial_matches(indicies, distances, exclude)

    if max_matches:
        indicies, distances = _filter_by_max_matches(indicies, distances, max_matches)

    if return_distance:
        return indicies, distances
    else:
        return indicies


@singleton
def paired_subsequence_match(
    y,
    x,
    threshold=None,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    max_matches=None,
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
        The distance threshold used to consider a subsequence matching. If no threshold
        is selected, `max_matches´ default to 10.

    dim : int, optional
        The dim to search for shapelets

     metric : str or callable, optional
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

    max_matches : int, optional
        Return the top `max_matches` matches below `threshold`.

        - If a `threshold` is given, the default behaviour is to return all matching
          indices in the order of occurrence .
        - If no `threshold` is given, the default behaviour is to return the top 10
          matching indicies ordered by distance
        - If both `threshold` and `max_matches` are given the top matches are returned
          ordered by distance.

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

    distance_measure = _SUBSEQUENCE_DISTANCE_MEASURE.get(metric, None)
    if distance_measure is None:
        raise ValueError("unsupported metric (%r)" % metric)
    metric_params = metric_params if metric_params is not None else {}

    if n_jobs is not None:
        warnings.warn("n_jobs is not yet supported.", UserWarning)

    if threshold is None:
        threshold = np.inf
        if max_matches is None:
            max_matches = 10

    if callable(threshold):
        threshold_fn = threshold

        def max_dist(d):
            return d <= threshold_fn(d)

        threshold = np.inf
    elif isinstance(threshold, str):
        threshold_fn = _THRESHOLD.get(threshold, None)
        if threshold_fn is None:
            raise ValueError("invalid threshold (%r)" % threshold)

        def max_dist(d):
            return d <= threshold_fn(d)

        threshold = np.inf
    elif not isinstance(threshold, numbers.Real):
        raise ValueError("invalid threshold (%r)" % threshold)
    else:
        max_dist = None

    indicies, distances = _distance._paired_subsequence_match(
        y,
        x,
        threshold,
        dim,
        distance_measure(**metric_params),
        n_jobs,
    )

    if max_dist is not None:
        indicies, distances = _filter_by_max_dist(indicies, distances, max_dist)

    if max_matches:
        indicies, distances = _filter_by_max_matches(indicies, distances, max_matches)

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
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The input data. y will be broadcast to the shape of x if possible.

    y : : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The input data

    dim : int, optional
        The dim to compute distance

     metric : str or callable, optional
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
        An array of shape (n_samples, )
    """
    x = check_array(x, allow_multivariate=True, dtype=np.double)
    y = check_array(y, allow_multivariate=True, dtype=np.double)
    y = np.broadcast_to(y, x.shape)
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
    y=None,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    n_jobs=None,
):
    """Compute the distance between subsequences and time series

    Parameters
    ----------
    x : ndarray of shape (x_samples, n_timestep) or (x_samples, n_dims, n_timestep),
    optional
        The input data

    y : : ndarray of shape (y_samples, n_timestep) or (y_samples, n_dims, n_timestep)
        The input data

    dim : int, optional
        The dim to compute distance

     metric : str or callable, optional # noqa: E501
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

    if y is None:
        y = x

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


@array_or_scalar(squeeze=True)
def matrix_profile(
    x,
    y=None,
    *,
    window=5,
    dim=0,
    exclude=None,
    n_jobs=-1,
    return_index=False,
):
    """Compute the matrix profile.

    - If only `x` is given, compute the similarity self-join of every subsequence in `x`
      of size `window` to its nearest neighbor in `x` excluding trivial matches according
      to the `exclude` parameter.
    - If both `x` and `y` are given, compute the similarity join of every subsequenec in
      `y` of size `window` to its nearest neighbor in `x` excluding matches according to
      the `exclude` parameter.


    Parameters
    ----------
    x : array-like of shape (n_timestep, ), (n_samples, xn_timestep) or (n_samples, n_dim, xn_timestep) # noqa E501
        The first time series

    y : array-like of shape (n_timestep, ), (n_samples, yn_timestep) or (n_samples, n_dim, yn_timestep), optional # noqa E501
        The optional second time series. y is broadcast to the shape of x if possible.

    window : int or float, optional
        The subsequence size, by default 5

        - if float, a fraction of `y.shape[-1]`
        - if int, the exact subsequence size

    dim : int, optional
        The dim to compute the matrix profile for, by default 0

    exclude : int or float, optional
        The size of the exclusion zone. The default exclusion zone is  0.2 for
        similarity self-join and 0.0 for similarity join.

        - if float, expressed as a fraction of the windows size
        - if int, exact size (0 >= exclude < window)

    n_jobs : int, optional
        The number of jobs to use when computing the

    return_index : bool, optional
        Return the matrix profile index

    Returns
    -------
    mp : ndarray of shape (profile_size, ) or (n_samples, profile_size)
        The matrix profile

    mpi : ndarray of shape (profile_size, ) or (n_samples, profile_size), optional
        The matrix profile index

    Notes
    -----
    The `profile_size` depends on the input.

    - If `y` is `None´, `profile_size` is  ``x.shape[-1] - window + 1``
    - If `y` is not `None`, `profile_size` is ``y.shape[-1] - window + 1``

    References
    ----------
    Yeh, C. C. M. et al. (2016).
        Matrix profile I: All pairs similarity joins for time series: a unifying view
        that includes motifs, discords and shapelets. In 2016 IEEE 16th international
        conference on data mining (ICDM)
    """
    x = check_array(np.atleast_2d(x), allow_multivariate=True)

    if y is not None:
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        y = np.broadcast_to(check_array(y, allow_multivariate=True), x.shape)

        if x.ndim != y.ndim:
            raise ValueError("both x and y must have the same dimensionality")
        if x.shape[0] != y.shape[0]:
            raise ValueError("both x and y must have the same number of samples")
        if x.ndim > 2 and x.shape[1] != y.shape[1]:
            raise ValueError("both x and y must have the same number of dimensions")
        if not y.shape[-1] <= x.shape[-1]:
            raise ValueError(
                "y.shape[-1] > x.shape[-1]. If you wan't to compute the matrix profile "
                "of the similarity join of YX, swap the order of inputs."
            )
        exclude = exclude or 0
    else:
        y = x
        exclude = exclude or 0.2

    if x.ndim > 2 and not 0 <= dim < x.shape[1]:
        raise ValueError("invalid dim (%d)" % x.shape[1])

    if isinstance(exclude, numbers.Real):
        exclude = math.ceil(window * exclude)
    elif exclude < 0:
        raise ValueError("invalid exclusion (%d < 0)" % exclude)

    if isinstance(window, numbers.Real):
        if not 0.0 < window <= 1.0:
            raise ValueError("invalid window size, got %f (expected [0, 1[)")
        window = math.ceil(window * y.shape[-1])
    elif window > y.shape[-1] or window > x.shape[-1] or window < 1:
        raise ValueError("invalid window size, got %r" % window)

    mp, mpi = _matrix_profile._paired_matrix_profile(
        x,
        y,
        window,
        dim,
        exclude,
        n_jobs,
    )

    if return_index:
        return mp, mpi
    else:
        return mp


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
