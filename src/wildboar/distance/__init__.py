# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers
import warnings

import numpy as np
from sklearn.utils.validation import _is_arraylike, check_scalar

from ..utils import _safe_jagged_array
from ..utils.validation import check_array, check_option, check_type
from . import _distance, _elastic, _mass, _matrix_profile, _metric

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
    "euclidean": _metric.EuclideanSubsequenceDistanceMeasure,
    "normalized_euclidean": _metric.NormalizedEuclideanSubsequenceDistanceMeasure,  # noqa: E501
    "scaled_euclidean": _metric.ScaledEuclideanSubsequenceDistanceMeasure,
    "dtw": _elastic.DtwSubsequenceDistanceMeasure,
    "scaled_dtw": _elastic.ScaledDtwSubsequenceDistanceMeasure,
    "mass": _mass.ScaledMassSubsequenceDistanceMeasure,
    "manhattan": _metric.ManhattanSubsequenceDistanceMeasure,
    "minkowski": _metric.MinkowskiSubsequenceDistanceMeasure,
    "chebyshev": _metric.ChebyshevSubsequenceDistanceMeasure,
    "cosine": _metric.CosineSubsequenceDistanceMeasure,
    "angular": _metric.AngularSubsequenceDistanceMeasure,
}

_DISTANCE_MEASURE = {
    "euclidean": _metric.EuclideanDistanceMeasure,
    "normalized_euclidean": _metric.NormalizedEuclideanDistanceMeasure,
    "dtw": _elastic.DtwDistanceMeasure,
    "ddtw": _elastic.DerivativeDtwDistanceMeasure,
    "wdtw": _elastic.WeightedDtwDistanceMeasure,
    "wddtw": _elastic.WeightedDerivativeDtwDistanceMeasure,
    "lcss": _elastic.LcssDistanceMeasure,
    "wlcss": _elastic.WeightedLcssDistanceMeasure,
    "erp": _elastic.ErpDistanceMeasure,
    "edr": _elastic.EdrDistanceMeasure,
    "msm": _elastic.MsmDistanceMeasure,
    "twe": _elastic.TweDistanceMeasure,
    "manhattan": _metric.ManhattanDistanceMeasure,
    "minkowski": _metric.MinkowskiDistanceMeasure,
    "chebyshev": _metric.ChebyshevDistanceMeasure,
    "cosine": _metric.CosineDistanceMeasure,
    "angular": _metric.AngularDistanceMeasure,
}

_THRESHOLD = {
    "best": lambda x: max(np.mean(x) - 2.0 * np.std(x), np.min(x)),
}


def _validate_subsequence(y):
    if isinstance(y, np.ndarray) and y.dtype != object:
        if y.ndim == 1:
            return [y.astype(float)]
        elif y.ndim == 2:
            y = list(y.astype(float))
        else:
            raise ValueError(
                "Expected 2D array, got {}D array instead:\narray={}.\n".format(
                    y.ndim, y
                )
            )
    else:
        if any(_is_arraylike(e) for e in y):
            y = [np.array(e, dtype=np.double) for e in y]
        else:
            y = [np.array(y, dtype=np.double)]

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


def _format_return(x, y_dims, x_dims):
    if x_dims == 1 and y_dims == 1 and x.size == 1:
        return x.item()
    elif x_dims == 1 or y_dims == 1:
        return np.squeeze(x)
    else:
        return x


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

    x : ndarray of shape (n_timestep, ), (n_samples, n_timestep)\
    or (n_samples, n_dims, n_timestep)
        The input data

    dim : int, optional
        The dim to search for subsequence

     metric : str or callable, optional
        The distance metric

        See ``_SUBSEQUENCE_DISTANCE_MEASURE.keys()`` for a list of supported metrics.

    metric_params: dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.

    return_index : bool, optional
        - if True return the index of the best match. If there are many equally good
          matches, the first match is returned.

    Returns
    -------
    dist : float, ndarray
        The minumum distance. Return depends on input:

        - if len(y) > 1 and x.ndim > 1, return an array of shape
          (n_samples, n_subsequences).
        - if len(y) == 1, return an array of shape (n_samples, ).
        - if x.ndim == 1, return an array of shape (n_subsequences, ).
        - if x.ndim == 1 and len(y) == 1, return scalar.

    indices : int, ndarray, optional
         The start index of the minumum distance. Return dependes on input:

        - if len(y) > 1 and x.ndim > 1, return an array of shape
          (n_samples, n_subsequences).
        - if len(y) == 1, return an array of shape (n_samples, ).
        - if x.ndim == 1, return an array of shape (n_subsequences, ).
        - if x.ndim == 1 and len(y) == 1, return scalar.
    """
    y = _validate_subsequence(y)
    x = check_array(x, allow_3d=True, ensure_2d=False, dtype=np.double)
    for s in y:
        if s.shape[0] > x.shape[-1]:
            raise ValueError(
                "Invalid subsequnce shape (%d > %d)" % (s.shape[0], x.shape[-1])
            )

    DistanceMeasure = check_option(_SUBSEQUENCE_DISTANCE_MEASURE, metric, "metric")
    metric_params = metric_params or {}
    min_dist, min_ind = _distance._pairwise_subsequence_distance(
        y,
        np.atleast_2d(x),
        dim,
        DistanceMeasure(**metric_params),
        n_jobs,
    )
    if return_index:
        return (
            _format_return(min_dist, len(y), x.ndim),
            _format_return(min_ind, len(y), x.ndim),
        )
    else:
        return _format_return(min_dist, len(y), x.ndim)


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
    y : list or ndarray of shape (n_samples, m_timestep)
        Input time series.

        - if list, a list of array-like of shape (m_timestep, )

    x : ndarray of shape (n_timestep, ), (n_samples, n_timestep)\
    or (n_samples, n_dims, n_timestep)
        The input data

    dim : int, optional
        The dim to search for shapelets

    metric : str or callable, optional
        The distance metric

        See ``_SUBSEQUENCE_DISTANCE_MEASURE.keys()`` for a list of supported metrics.

    metric_params: dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.

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

    indices : int, ndarray, optional
        An array of shape (n_samples, ) with the index of the best matching position
        of the i:th subsequence and the i:th sample
    """
    y = _validate_subsequence(y)
    x = check_array(x, allow_3d=True, ensure_2d=False, dtype=np.double)
    for s in y:
        if s.shape[0] > x.shape[-1]:
            raise ValueError(
                "Invalid subsequnce shape (%d > %d)" % (s.shape[0], x.shape[-1])
            )
    DistanceMeasure = check_option(_SUBSEQUENCE_DISTANCE_MEASURE, metric, "metric")
    if n_jobs is not None:
        warnings.warn("n_jobs is not yet supported.", UserWarning)

    n_samples = x.shape[0] if x.ndim > 1 else 1
    if len(y) != n_samples:
        raise ValueError(
            "The number of subsequences and samples must be the same, got %d "
            "subsequences and %d samples." % (len(y), n_samples)
        )

    metric_params = metric_params if metric_params is not None else {}
    min_dist, min_ind = _distance._paired_subsequence_distance(
        y, np.atleast_2d(x), dim, DistanceMeasure(**metric_params)
    )
    if return_index:
        return (
            _format_return(min_dist, len(y), x.ndim),
            _format_return(min_ind, len(y), x.ndim),
        )
    else:
        return _format_return(min_dist, len(y), x.ndim)


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

    x : ndarray of shape (n_timestep, ), (n_samples, n_timestep)\
    or (n_samples, n_dims, n_timestep)
        The input data

    threshold : str, float or callable, optional
        The distance threshold used to consider a subsequence matching. If no threshold
        is selected, `max_matches` defaults to 10.

        - if float, return all matches closer than threshold
        - if callable, return all matches closer than the treshold computed by the
          threshold function, given all distances to the subsequence
        - if str, return all matches according to the named threshold.

    dim : int, optional
        The dim to search for shapelets

    metric : str or callable, optional
        The distance metric

        See ``_SUBSEQUENCE_DISTANCE_MEASURE.keys()`` for a list of supported metrics.

    metric_params: dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.

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
    indicies : ndarray
        The start index of matching subsequences. Return depends on input:

        - if x.ndim > 1, return an ndarray of shape (n_samples, )
        - if x.ndim == 1, return ndarray of shape (n_matches, ) or None

        For each sample, the ndarray contains the .

    distance : ndarray, optional
        The distances of matching subsequences. Return depends on input:

        - if x.ndim > 1, return an ndarray of shape (n_samples, )
        - if x.ndim == 1, return ndarray of shape (n_matches, ) or None
    """
    y = _validate_subsequence(y)
    if len(y) > 1:
        raise ValueError("A single subsequence expected, got %d" % len(y))
    y = y[0]
    x = check_array(x, allow_3d=True, ensure_2d=False, dtype=np.double)
    if y.shape[0] > x.shape[-1]:
        raise ValueError(
            "Invalid subsequnce shape (%d > %d)" % (y.shape[0], x.shape[-1])
        )
    DistanceMeasure = check_option(_SUBSEQUENCE_DISTANCE_MEASURE, metric, "metric")
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
        threshold_fn = check_option(_THRESHOLD, threshold, "threshold")

        def max_dist(d):
            return d <= threshold_fn(d)

        threshold = np.inf
    elif not isinstance(threshold, numbers.Real):
        raise TypeError(
            "threshold must be str, callable or float, not %s"
            % type(threshold).__qualname__
        )
    else:
        max_dist = None

    check_type(exclude, "exclude", (numbers.Integral, numbers.Real), required=False)
    if isinstance(exclude, numbers.Integral):
        check_scalar(exclude, "exclude", numbers.Integral, min_val=0)
    elif isinstance(exclude, numbers.Real):
        check_scalar(
            exclude,
            "exclude",
            numbers.Real,
            min_val=0,
        )
        exclude = math.ceil(y.size * exclude)

    indicies, distances = _distance._subsequence_match(
        y,
        np.atleast_2d(x),
        threshold,
        dim,
        DistanceMeasure(**metric_params),
        n_jobs,
    )

    if max_dist is not None:
        indicies, distances = _filter_by_max_dist(indicies, distances, max_dist)

    if exclude:
        indicies, distances = _exclude_trivial_matches(indicies, distances, exclude)

    if max_matches:
        indicies, distances = _filter_by_max_matches(indicies, distances, max_matches)

    if return_distance:
        return (
            _format_return(_safe_jagged_array(indicies), 1, x.ndim),
            _format_return(_safe_jagged_array(distances), 1, x.ndim),
        )
    else:
        return _format_return(_safe_jagged_array(indicies), 1, x.ndim)


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

        - if list, a list of array-like of shape (n_timestep, ) with length n_samples

    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The input data

    threshold : float
        The distance threshold used to consider a subsequence matching. If no threshold
        is selected, `max_matches` defaults to 10.

    dim : int, optional
        The dim to search for shapelets

    metric : str or callable, optional
        The distance metric

        See ``_SUBSEQUENCE_DISTANCE_MEASURE.keys()`` for a list of supported metrics.

    metric_params: dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.

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
    indicies : ndarray
        The start index of matching subsequences. Return depends on input:

        - if x.ndim > 1, return an ndarray of shape (n_samples, )
        - if x.ndim == 1, return ndarray of shape (n_matches, ) or None

        For each sample, the ndarray contains the .

    distance : ndarray, optional
        The distances of matching subsequences. Return depends on input:

        - if x.ndim > 1, return an ndarray of shape (n_samples, )
        - if x.ndim == 1, return ndarray of shape (n_matches, ) or None
    """
    y = _validate_subsequence(y)
    x = check_array(x, allow_3d=True, dtype=np.double)
    if len(y) != x.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    for s in y:
        if s.shape[0] > x.shape[-1]:
            raise ValueError(
                "invalid subsequnce shape (%d > %d)" % (s.shape[0], x.shape[-1])
            )

    DistanceMeasure = check_option(_SUBSEQUENCE_DISTANCE_MEASURE, metric, "metric")
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
        threshold_fn = check_option(_THRESHOLD, threshold, "threshold")

        def max_dist(d):
            return d <= threshold_fn(d)

        threshold = np.inf
    elif not isinstance(threshold, numbers.Real):
        raise TypeError(
            "threshold must be str, callable or float, not %s"
            % type(threshold).__qualname__
        )
    else:
        max_dist = None

    indicies, distances = _distance._paired_subsequence_match(
        y,
        x,
        threshold,
        dim,
        DistanceMeasure(**metric_params),
        n_jobs,
    )

    if max_dist is not None:
        indicies, distances = _filter_by_max_dist(indicies, distances, max_dist)

    if max_matches:
        indicies, distances = _filter_by_max_matches(indicies, distances, max_matches)

    if return_distance:
        return (
            _format_return(_safe_jagged_array(indicies), len(y), x.ndim),
            _format_return(_safe_jagged_array(distances), len(y), x.ndim),
        )
    else:
        return _format_return(_safe_jagged_array(indicies), len(y), x.ndim)


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

        See ``_DISTANCE_MEASURE.keys()`` for a list of supported metrics.

    metric_params: dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.

    n_jobs : int, optional
        The number of parallel jobs.

    Returns
    -------
    distance : ndarray
        The distances. Return depends on input:

        - if ndim > 1, return an ndarray of shape (n_samples, )
        - if ndim == 1, return ndarray of shape (n_matches, ) or None
    """
    x = check_array(x, allow_3d=True, ensure_2d=False, dtype=np.double)
    y = check_array(y, allow_3d=True, ensure_2d=False, dtype=np.double)
    y = np.broadcast_to(y, x.shape)
    if x.ndim != y.ndim:
        raise ValueError(
            "x (%dD-array) and y (%dD-array) are not compatible." % (x.ndim, y.ndim)
        )
    if x.ndim > 1 and y.ndim > 1 and x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples.")

    if n_jobs is not None:
        warnings.warn("n_jobs is not yet supported.", UserWarning)

    DistanceMeasure = check_option(_DISTANCE_MEASURE, metric, "metric")
    metric_params = metric_params if metric_params is not None else {}
    distance_measure = DistanceMeasure(**metric_params)
    if x.shape[x.ndim - 1] != x.shape[x.ndim - 1] and not distance_measure.is_elastic:
        raise ValueError(
            "Illegal n_timestep (%r != %r) for non-elastic distance measure"
            % (x.shape[x.ndim - 1], y.shape[y.ndim - 1])
        )

    return _format_return(
        _distance._paired_distance(
            np.atleast_2d(x),
            np.atleast_2d(y),
            dim,
            distance_measure,
            n_jobs,
        ),
        y.ndim,
        x.ndim,
    )


def mean_paired_distance(x, y, *, metric="euclidean", metric_params=None):
    """Return the paired distance between x and y. For 3d-arrays, return the mean
    paired distance between the dimensions.

    Parameters
    ----------


    """
    x = check_array(x, allow_3d=True, input_name="x")
    y = check_array(y, allow_3d=True, input_name="y")

    if x.ndim != y.ndim:
        raise ValueError("both x and y must have the same rank")

    if x.ndim == 2:
        return paired_distance(x, y, metric=metric, metric_params=metric_params)
    elif x.ndim == 3:
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                "x.shape[1]=%d != y.shape[1]=%d" % (x.shape[1], y.shape[1])
            )

        return np.mean(
            [
                paired_distance(
                    x, y, metric=metric, metric_params=metric_params, dim=dim
                )
                for dim in range(x.shape[1])
            ],
            axis=0,
        )
    else:
        raise ValueError("invalid rank, %d > 3" % x.ndim)


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
    x : ndarray of shape (n_timestep, ), (x_samples, n_timestep) or \
            (x_samples, n_dims, n_timestep)
        The input data

    y : ndarray of shape (n_timestep, ), (y_samples, n_timestep) or \
            (y_samples, n_dims, n_timestep), optional
        The input data

    dim : int, optional
        The dim to compute distance

     metric : str or callable, optional
        The distance metric

        See ``_DISTANCE_MEASURE.keys()`` for a list of supported metrics.

    metric_params: dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.

    n_jobs : int, optional
        The number of parallel jobs.

    Returns
    -------
    dist : float or ndarray
        The distances. Return depends on input.

        - if x.ndim > 1 and y is None, return array of shape (x_samples, x_samples)
        - if x.ndim > 1 and y.ndim > 1, return array of shape (x_samples, y_samples)
        - if x.ndim == 1 and y.ndim > 1, return array of shape (y_samples, )
        - if y.ndim == 1 and x.ndim > 1, return array of shape (x_samples, )
        - if x.ndim == 1 and y.ndim == 1, return scalar
    """
    DistanceMeasure = check_option(_DISTANCE_MEASURE, metric, "metric")
    metric_params = metric_params if metric_params is not None else {}
    distance_measure = DistanceMeasure(**metric_params)

    if y is None:
        y = x

    if x is y:
        x = check_array(x, allow_3d=True, ensure_2d=False, dtype=np.double)
        if not 0 >= dim < x.ndim:
            raise ValueError("dim must be 0 <= %d < %d" % (dim, x.ndim))

        if x.ndim == 1:
            return 0.0

        return _distance._singleton_pairwise_distance(x, dim, distance_measure, n_jobs)
    else:
        x = check_array(x, allow_3d=True, ensure_2d=False, dtype=np.double)
        y = check_array(y, allow_3d=True, ensure_2d=False, dtype=np.double)
        if x.ndim != 1 and y.ndim != 1 and x.ndim != y.ndim:
            raise ValueError(
                "x (%dD-array) and y (%dD-array) are not compatible" % (x.ndim, y.ndim)
            )

        if not 0 >= dim < x.ndim:
            raise ValueError("dim must be 0 <= %d < %d" % (dim, x.ndim))

        if (
            x.shape[x.ndim - 1] != y.shape[y.ndim - 1]
            and not distance_measure.is_elastic
        ):
            raise ValueError(
                "Illegal n_timestep (%r != %r) for non-elastic distance measure"
                % (x.shape[x.ndim - 1], y.shape[y.ndim - 1])
            )

        return _format_return(
            _distance._pairwise_distance(
                np.atleast_2d(x),
                np.atleast_2d(y),
                dim,
                distance_measure,
                n_jobs,
            ),
            y.ndim,
            x.ndim,
        )


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

    - If only ``x`` is given, compute the similarity self-join of every subsequence in
      ``x`` of size ``window`` to its nearest neighbor in `x` excluding trivial matches
      according to the ``exclude`` parameter.
    - If both ``x`` and ``y`` are given, compute the similarity join of every
      subsequenec in ``y`` of size ``window`` to its nearest neighbor in ``x`` excluding
      matches according to the ``exclude`` parameter.

    Parameters
    ----------
    x : array-like of shape (n_timestep, ), (n_samples, xn_timestep) or \
        (n_samples, n_dim, xn_timestep)
        The first time series

    y : array-like of shape (n_timestep, ), (n_samples, yn_timestep) or \
        (n_samples, n_dim, yn_timestep), optional
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
    x = check_array(x, allow_3d=True, ensure_2d=False)

    if y is not None:
        y = check_array(y, allow_3d=True, ensure_2d=False)
        if x.ndim > 1:
            y = np.broadcast_to(y, x.shape)

        if x.ndim != y.ndim:
            raise ValueError("Both x and y must have the same dimensionality")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Both x and y must have the same number of samples")
        if x.ndim > 2 and x.shape[1] != y.shape[1]:
            raise ValueError("Both x and y must have the same number of dimensions")
        if not y.shape[-1] <= x.shape[-1]:
            raise ValueError(
                "y.shape[-1] > x.shape[-1]. If you want to compute the matrix profile "
                "of the similarity join of YX, swap the order of inputs."
            )
        exclude = exclude if exclude is not None else 0.0
    else:
        y = x
        exclude = exclude if exclude is not None else 0.2

    if x.ndim > 2 and not 0 <= dim < x.shape[1]:
        raise ValueError("Invalid dim (%d)" % x.shape[1])

    check_type(window, "window", (numbers.Integral, numbers.Real))
    check_type(exclude, "exclude", (numbers.Integral, numbers.Real))
    if isinstance(window, numbers.Integral):
        check_scalar(
            window,
            "window",
            numbers.Integral,
            min_val=1,
            max_val=min(y.shape[-1], x.shape[-1]),
        )
    elif isinstance(window, numbers.Real):
        check_scalar(
            window,
            "window",
            numbers.Real,
            min_val=0,
            max_val=1,
            include_boundaries="right",
        )
        window = math.ceil(window * y.shape[-1])

    if isinstance(exclude, numbers.Integral):
        check_scalar(exclude, "exclude", numbers.Integral, min_val=1)
    elif isinstance(exclude, numbers.Real):
        check_scalar(
            exclude,
            "exclude",
            numbers.Real,
            min_val=0,
        )
        exclude = math.ceil(window * exclude)

    mp, mpi = _matrix_profile._paired_matrix_profile(
        np.atleast_2d(x),
        np.atleast_2d(y),
        window,
        dim,
        exclude,
        n_jobs,
    )

    if return_index:
        return _format_return(mp, 2, x.ndim), _format_return(mpi, 2, x.ndim)
    else:
        return _format_return(mp, 2, x.ndim)
