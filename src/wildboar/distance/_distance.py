import math
import numbers
import warnings

import numpy as np
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
    validate_params,
)
from sklearn.utils.validation import _is_arraylike, check_scalar

from ..utils import _safe_jagged_array
from ..utils.validation import _check_ts_array, check_array, check_option, check_type
from ._cdistance import (
    _argmin_distance,
    _paired_distance,
    _paired_subsequence_distance,
    _paired_subsequence_match,
    _pairwise_distance,
    _pairwise_subsequence_distance,
    _singleton_pairwise_distance,
    _subsequence_distance_profile,
    _subsequence_match,
)
from ._elastic import (
    AmercingDtwMetric,
    AmercingDtwSubsequenceMetric,
    DerivativeDtwMetric,
    DerivativeDtwSubsequenceMetric,
    DtwMetric,
    DtwSubsequenceMetric,
    EdrMetric,
    EdrSubsequenceMetric,
    ErpMetric,
    ErpSubsequenceMetric,
    LcssMetric,
    LcssSubsequenceMetric,
    MsmMetric,
    MsmSubsequenceMetric,
    ScaledDtwSubsequenceMetric,
    TweMetric,
    TweSubsequenceMetric,
    WeightedDerivativeDtwMetric,
    WeightedDerivativeDtwSubsequenceMetric,
    WeightedDtwMetric,
    WeightedDtwSubsequenceMetric,
    WeightedLcssMetric,
)
from ._mass import ScaledMassSubsequenceMetric

# from . import _cdistance, _elastic, _mass, _metric
from ._metric import (
    AngularMetric,
    AngularSubsequenceMetric,
    ChebyshevMetric,
    ChebyshevSubsequenceMetric,
    CosineMetric,
    CosineSubsequenceMetric,
    EuclideanMetric,
    EuclideanSubsequenceMetric,
    ManhattanMetric,
    ManhattanSubsequenceMetric,
    MinkowskiMetric,
    MinkowskiSubsequenceMetric,
    NormalizedEuclideanMetric,
    NormalizedEuclideanSubsequenceMetric,
    ScaledEuclideanSubsequenceMetric,
)

_SUBSEQUENCE_METRICS = {
    "euclidean": EuclideanSubsequenceMetric,
    "normalized_euclidean": NormalizedEuclideanSubsequenceMetric,
    "scaled_euclidean": ScaledEuclideanSubsequenceMetric,
    "adtw": AmercingDtwSubsequenceMetric,
    "dtw": DtwSubsequenceMetric,
    "wdtw": WeightedDtwSubsequenceMetric,
    "ddtw": DerivativeDtwSubsequenceMetric,
    "wddtw": WeightedDerivativeDtwSubsequenceMetric,
    "scaled_dtw": ScaledDtwSubsequenceMetric,
    "lcss": LcssSubsequenceMetric,
    "edr": EdrSubsequenceMetric,
    "twe": TweSubsequenceMetric,
    "msm": MsmSubsequenceMetric,
    "erp": ErpSubsequenceMetric,
    "mass": ScaledMassSubsequenceMetric,
    "manhattan": ManhattanSubsequenceMetric,
    "minkowski": MinkowskiSubsequenceMetric,
    "chebyshev": ChebyshevSubsequenceMetric,
    "cosine": CosineSubsequenceMetric,
    "angular": AngularSubsequenceMetric,
}

_METRICS = {
    "euclidean": EuclideanMetric,
    "normalized_euclidean": NormalizedEuclideanMetric,
    "adtw": AmercingDtwMetric,
    "dtw": DtwMetric,
    "ddtw": DerivativeDtwMetric,
    "wdtw": WeightedDtwMetric,
    "wddtw": WeightedDerivativeDtwMetric,
    "lcss": LcssMetric,
    "wlcss": WeightedLcssMetric,
    "erp": ErpMetric,
    "edr": EdrMetric,
    "msm": MsmMetric,
    "twe": TweMetric,
    "manhattan": ManhattanMetric,
    "minkowski": MinkowskiMetric,
    "chebyshev": ChebyshevMetric,
    "cosine": CosineMetric,
    "angular": AngularMetric,
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
    elif any(_is_arraylike(e) for e in y):
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
    """Minimum subsequence distance between subsequences and time series.

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

        See ``_SUBSEQUENCE_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
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

    Metric = check_option(_SUBSEQUENCE_METRICS, metric, "metric")  # noqa: N806
    metric_params = metric_params or {}
    min_dist, min_ind = _pairwise_subsequence_distance(
        y,
        _check_ts_array(x),
        dim,
        Metric(**metric_params),
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
    """Minimum subsequence distance between the i:th subsequence and time series.

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

        See ``_SUBSEQUENCE_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
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
    x = check_array(x, allow_3d=True, ensure_2d=False, dtype=float)

    n_dims = x.shape[1] if x.ndim == 3 else 1
    if not 0 >= dim < n_dims:
        raise ValueError("The parameter dim must be 0 <= dim < n_dims")

    for s in y:
        if s.shape[0] > x.shape[-1]:
            raise ValueError(
                "Invalid subsequnce shape (%d > %d)" % (s.shape[0], x.shape[-1])
            )

    Metric = check_option(_SUBSEQUENCE_METRICS, metric, "metric")  # noqa: N806
    if n_jobs is not None:
        warnings.warn("n_jobs is not yet supported.", UserWarning)

    n_samples = x.shape[0] if x.ndim > 1 else 1
    if len(y) != n_samples:
        raise ValueError(
            "The number of subsequences and samples must be the same, got %d "
            "subsequences and %d samples." % (len(y), n_samples)
        )

    metric_params = metric_params if metric_params is not None else {}
    min_dist, min_ind = _paired_subsequence_distance(
        y, _check_ts_array(x), dim, Metric(**metric_params)
    )
    if return_index:
        return (
            _format_return(min_dist, len(y), x.ndim),
            _format_return(min_ind, len(y), x.ndim),
        )
    else:
        return _format_return(min_dist, len(y), x.ndim)


def subsequence_match(  # noqa: PLR0912
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
    """Find matching subsequnces.

    Find the positions where the distance is less than the threshold between
    the subsequence and all time series.

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

        See ``_SUBSEQUENCE_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
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
    x = check_array(x, allow_3d=True, ensure_2d=False, dtype=float)

    if y.shape[0] > x.shape[-1]:
        raise ValueError(
            "Invalid subsequnce shape (%d > %d)" % (y.shape[0], x.shape[-1])
        )

    n_dims = x.shape[1] if x.ndim == 3 else 1
    if not 0 >= dim < n_dims:
        raise ValueError("The parameter dim must be 0 <= dim < n_dims")

    Metric = check_option(_SUBSEQUENCE_METRICS, metric, "metric")  # noqa: N806
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

    indicies, distances = _subsequence_match(
        y,
        _check_ts_array(x),
        threshold,
        dim,
        Metric(**metric_params),
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


def paired_subsequence_match(  # noqa: PLR0912
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
    """Find matching subsequnces.

    Find the positions where the distance is less than the threshold between
    the i:th subsequences and time series.

    - If a `threshold` is given, the default behaviour is to return all matching
      indices in the order of occurrence
    - If no `threshold` is given, the default behaviour is to return the top 10
      matching indicies ordered by distance
    - If both `threshold` and `max_matches` are given, the top matches are returned
      ordered by distance and time series.

    Parameters
    ----------
    y : list or ndarray of shape (n_samples, n_timestep)
        Input time series.

        - if list, a list of array-like of shape (n_timestep, ) with length n_samples.

    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The input data.
    threshold : float, optional
        The distance threshold used to consider a subsequence matching. If no threshold
        is selected, `max_matches` defaults to 10.
    dim : int, optional
        The dim to search for shapelets.
    metric : str or callable, optional
        The distance metric

        See ``_SUBSEQUENCE_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
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
        If True, return the distance of the match.
    n_jobs : int, optional
        The number of parallel jobs to run. Ignored.

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
        - if x.ndim == 1, return ndarray of shape (n_matches, ) or None.
    """
    y = _validate_subsequence(y)
    x = check_array(x, allow_3d=True, dtype=np.double)
    if len(y) != x.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    n_dims = x.shape[1] if x.ndim == 3 else 1
    if not 0 >= dim < n_dims:
        raise ValueError("The parameter dim must be 0 <= dim < n_dims")

    for s in y:
        if s.shape[0] > x.shape[-1]:
            raise ValueError(
                "invalid subsequnce shape (%d > %d)" % (s.shape[0], x.shape[-1])
            )

    Metric = check_option(_SUBSEQUENCE_METRICS, metric, "metric")  # noqa: N806
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

    indicies, distances = _paired_subsequence_match(
        y,
        _check_ts_array(x),
        threshold,
        dim,
        Metric(**metric_params),
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


def paired_distance(  # noqa: PLR0912
    x,
    y,
    *,
    dim="warn",
    metric="euclidean",
    metric_params=None,
    n_jobs=None,
):
    """Compute the distance between the i:th time series.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The input data.
    y : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timestep)
        The input data. y will be broadcasted to the shape of x.
    dim : int or {'mean', 'full'}, optional
        The dim to compute distance.
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.
    n_jobs : int, optional
        The number of parallel jobs.

    Returns
    -------
    ndarray
        The distances. Return depends on input:

        - if x.ndim == 1, return scalar.
        - if dim='full', return ndarray of shape (n_dims, n_samples).
        - if x.ndim > 1, return an ndarray of shape (n_samples, ).
    """
    x = check_array(x, allow_3d=True, ensure_2d=False, dtype=float)
    y = check_array(y, allow_3d=True, ensure_2d=False, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    if x.ndim != y.ndim:
        raise ValueError(
            "x (%dD-array) and y (%dD-array) are not compatible." % (x.ndim, y.ndim)
        )

    if x.ndim == 3 and x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same number of dimensions.")

    if x.ndim > 1 and y.ndim > 1 and x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples.")

    if n_jobs is not None:
        warnings.warn("n_jobs is not yet supported.", UserWarning)

    Metric = check_option(_METRICS, metric, "metric")  # noqa: N806
    metric_params = metric_params if metric_params is not None else {}
    metric = Metric(**metric_params)
    if x.shape[x.ndim - 1] != x.shape[x.ndim - 1] and not metric.is_elastic:
        raise ValueError(
            "Illegal n_timestep (%r != %r) for non-elastic distance measure"
            % (x.shape[x.ndim - 1], y.shape[y.ndim - 1])
        )

    n_dims = x.shape[1] if x.ndim == 3 else 1

    # TODO(1.3)
    if dim == "warn":
        if n_dims > 1:
            warnings.warn(
                "The default value for dim will change to 'mean' from 0 in 1.3. "
                "Explicitly set dim=0 to keep the current behaviour for 3d-arrays.",
                DeprecationWarning,
            )
        dim = 0

    if n_dims == 1 and dim == "mean":
        dim = 0

    x_ = _check_ts_array(x)
    y_ = _check_ts_array(y)
    if dim in ["mean", "full"]:
        distances = [_paired_distance(x_, y_, d, metric, n_jobs) for d in range(n_dims)]

        if dim == "mean":
            distances = np.mean(distances, axis=0)
        else:
            distances = np.stack(distances, axis=0)

    elif isinstance(dim, numbers.Integral) and 0 <= dim < n_dims:
        distances = _paired_distance(x_, y_, dim, metric, n_jobs)
    else:
        raise ValueError("The parameter dim must be 0 <= dim < n_dims")

    return _format_return(distances, y.ndim, x.ndim)


@np.deprecate(new_name="paired_distance(dim='mean')")
def mean_paired_distance(x, y, *, metric="euclidean", metric_params=None):
    """ """
    return paired_distance(x, y, dim="mean", metric=metric, metric_params=metric_params)


def pairwise_distance(  # noqa: PLR
    x,
    y=None,
    *,
    dim="warn",
    metric="euclidean",
    metric_params=None,
    n_jobs=None,
):
    """
    Compute the distance between subsequences and time series.

    Parameters
    ----------
    x : ndarray of shape (n_timestep, ), (x_samples, n_timestep) or \
            (x_samples, n_dims, n_timestep)
        The input data.
    y : ndarray of shape (n_timestep, ), (y_samples, n_timestep) or \
            (y_samples, n_dims, n_timestep), optional
        The input data.
    dim : int or {'mean', 'full'}, optional
        The dim to compute distance.
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.
    n_jobs : int, optional
        The number of parallel jobs.

    Returns
    -------
    float or ndarray
        The distances. Return depends on input.

        - if x.ndim == 1 and y.ndim == 1, scalar.
        - if dim="full", array of shape (n_dims, x_samples, y_samples).
        - if dim="full" and y is None, array of shape (n_dims, x_samples, x_samples).
        - if x.ndim > 1 and y is None, array of shape (x_samples, x_samples).
        - if x.ndim > 1 and y.ndim > 1, array of shape (x_samples, y_samples).
        - if x.ndim == 1 and y.ndim > 1, array of shape (y_samples, ).
        - if y.ndim == 1 and x.ndim > 1, array of shape (x_samples, ).

    """
    Metric = check_option(_METRICS, metric, "metric")  # noqa: N806
    metric_params = metric_params if metric_params is not None else {}
    metric = Metric(**metric_params)

    if y is None:
        y = x

    if x is y:
        x = check_array(x, allow_3d=True, ensure_2d=False, dtype=float)
        if x.ndim == 1:
            return 0.0

        x_ = _check_ts_array(x)
        n_dims = x.shape[1] if x.ndim == 3 else 1

        # TODO(1.3)
        if dim == "warn":
            if n_dims > 1:
                warnings.warn(
                    "The default value for dim will change to 'mean' from 0 in 1.3. "
                    "Explicitly set dim=0 to keep the current behaviour for 3d-arrays.",
                    DeprecationWarning,
                )
            dim = 0

        if n_dims == 1 and dim == "mean":
            dim = 0

        if dim in ["mean", "full"]:
            distances = [
                _singleton_pairwise_distance(x_, d, metric, n_jobs)
                for d in range(n_dims)
            ]

            if dim == "mean":
                distances = np.mean(distances, axis=0)
            else:
                distances = np.stack(distances, axis=0)

        elif isinstance(dim, numbers.Integral) and 0 <= dim < n_dims:
            distances = _singleton_pairwise_distance(x_, dim, metric, n_jobs)
        else:
            raise ValueError("The parameter dim must be 0 <= dim < n_dims")

        return distances
    else:
        x = check_array(x, allow_3d=True, ensure_2d=False, dtype=np.double)
        y = check_array(y, allow_3d=True, ensure_2d=False, dtype=np.double)
        if x.ndim != 1 and y.ndim != 1 and x.ndim != y.ndim:
            raise ValueError(
                "x (%dD-array) and y (%dD-array) are not compatible" % (x.ndim, y.ndim)
            )

        if x.ndim == 3 and x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have the same number of dimensions.")

        if x.shape[-1] != y.shape[-1] and not metric.is_elastic:
            raise ValueError(
                "Illegal n_timestep (%r != %r) for non-elastic distance measure"
                % (x.shape[-1], y.shape[-1])
            )

        x_ = _check_ts_array(x)
        y_ = _check_ts_array(y)
        n_dims = x.shape[1] if x.ndim == 3 else 1

        # TODO(1.3)
        if dim == "warn":
            if n_dims > 1:
                warnings.warn(
                    "The default value for dim will change to 'mean' from 0 in 1.3. "
                    "Explicitly set dim=0 to keep the current behaviour for 3d-arrays.",
                    DeprecationWarning,
                )
            dim = 0

        if n_dims == 1 and dim == "mean":
            dim = 0

        if dim in ["mean", "full"]:
            distances = [
                _pairwise_distance(x_, y_, d, metric, n_jobs) for d in range(n_dims)
            ]

            if dim == "mean":
                distances = np.mean(distances, axis=0)
            else:
                distances = np.stack(distances, axis=0)

        elif isinstance(dim, numbers.Integral) and 0 <= dim < n_dims:
            distances = _pairwise_distance(x_, y_, dim, metric, n_jobs)
        else:
            raise ValueError("The parameter dim must be 0 <= dim < n_dims")

        return _format_return(distances, y.ndim, x.ndim)


@validate_params(
    {
        "x": ["array-like"],
        "y": [None, "array-like"],
        "k": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric": [StrOptions(_METRICS.keys())],
        "metric_params": [None, dict],
        "return_distance": [bool],
        "sorted": [bool],
        "n_jobs": [numbers.Integral, None],
    },
    prefer_skip_nested_validation=True,
)
def argmin_distance(
    x,
    y=None,
    *,
    dim=0,
    k=1,
    metric="euclidean",
    metric_params=None,
    sorted=False,
    return_distance=False,
    n_jobs=None,
):
    """
    Find the indicies of the samples with the lowest distance in `Y`.

    Parameters
    ----------
    x : univariate time-series or multivariate time-series
        The needle.
    y : univariate time-series or multivariate time-series, optional
        The haystack.
    dim : int, optional
        The dimension where the distance is computed.
    k : int, optional
        The number of closest samples.
    metric : str, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.
    sorted : bool, optional
        Sort the indicies from smallest to largest distance.
    return_distance : bool, optional
        Return the distance for the `k` samples.
    n_jobs : int, optional
        The number of parallel jobs.

    Returns
    -------
    indices : ndarray of shape (n_samples, k)
        The indices of the samples in `Y` with the smallest distance.
    distance : ndarray of shape (n_samples, k), optional
        The distance of the samples in `Y` with the smallest distance.

    Examples
    --------
    >>> from wildoar.distance import argmin_distance
    >>> X = np.array([[1, 2, 3, 4], [10, 1, 2, 3]])
    >>> Y = np.array([[1, 2, 11, 2], [2, 4, 6, 7], [10, 11, 2, 3]])
    >>> argmin_distance(X, Y, k=2, return_distance=True)
    (array([[0, 1],
            [1, 2]]),
     array([[ 8.24621125,  4.79583152],
            [10.24695077, 10.        ]]))
    """
    metric_params = metric_params if metric_params is not None else {}
    metric = _METRICS[metric](**metric_params)

    x = check_array(
        x, allow_3d=True, ensure_2d=False, ensure_ts_array=True, dtype=float
    )
    if y is None:
        y = x
    else:
        y = check_array(
            y, allow_3d=True, ensure_2d=False, ensure_ts_array=True, dtype=float
        )

    if x.ndim not in (1, y.ndim):
        raise ValueError(
            f"x ({x.ndim}d-array) and y ({y.ndim}d-array) are not compatible."
        )

    if x.shape[-1] != y.shape[-1] and not metric.is_elastic:
        raise ValueError(
            "Illegal n_timestep (%d != %d) for non-elastic distance measure."
            % (x.shape[-1], y.shape[-1])
        )

    n_dims = x.shape[1] if x.ndim == 3 else 1
    k = min(k, y.shape[0])
    if 0 <= dim < 1:
        indices, distances = _argmin_distance(x, y, dim, metric, k, n_jobs)

        if sorted:
            sort = np.argsort(distances, axis=1)
            indices = np.take_along_axis(indices, sort, axis=1)
            if return_distance:
                distances = np.take_along_axis(distances, sort, axis=1)

        if return_distance:
            return indices, distances
        else:
            return indices
    else:
        raise ValueError(f"The parameter dim must be dim ({dim}) < n_dims ({n_dims})")


def distance_profile(y, x, *, dim=0, metric="mass", metric_params=None, n_jobs=None):
    """
    Compute the distance profile.

    The distance profile of shape `(n_samples, n_timestep - yn_timestep + 1)`
    corresponds to the distance of the subsequence y for every time point
    in x.

    Parameters
    ----------
    y : array-like of shape (yn_timestep, )
        The subsequence.
    x : ndarray of shape (n_timestep, ), (n_samples, n_timestep)\
    or (n_samples, n_dims, n_timestep)
        The input data.
    dim : int, optional
        The dim to search for shapelets.
    metric : str or callable, optional
        The distance metric

        See ``_SUBSEQUENCE_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.
    n_jobs : int, optional
        The number of parallel jobs to run.

    Returns
    -------
    ndarray of shape (n_samples, n_timestep - yn_timestep + 1) or\
            (n_timestep - yn_timestep + 1, )
        The distance between every subsequence in `x` to `y`.

    Examples
    --------
    >>> from wildboar.datasets import load_dataset
    >>> from wildboar.distance import distance_profile
    >>> X, _ = load_dataset("ECG200")
    >>> distance_profile(X[0], X[1:].reshape(-1))
    array([14.00120332, 14.41943788, 14.81597243, ...,  4.75219094,
           5.72681005,  6.70155561])
    """
    y = _validate_subsequence(y)
    if len(y) > 1:
        raise ValueError("A single subsequence expected, got %d" % len(y))

    y = y[0]
    x = check_array(x, allow_3d=True, ensure_2d=False, dtype=float)

    if y.shape[0] > x.shape[-1]:
        raise ValueError(
            "Invalid subsequnce shape (%d > %d)" % (y.shape[0], x.shape[-1])
        )

    Metric = _SUBSEQUENCE_METRICS[metric]
    metric_params = metric_params if metric_params is not None else {}

    dp = _subsequence_distance_profile(
        y, _check_ts_array(x), dim, Metric(**metric_params), n_jobs
    )

    if dp.shape[0] == 1:
        return dp[0]
    else:
        return dp
