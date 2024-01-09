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
    CallableMetric,
    ScaledSubsequenceMetricWrap,
    SubsequenceMetricWrap,
    _argmin_distance,
    _argmin_subsequence_distance,
    _dilated_distance_profile,
    _distance_profile,
    _paired_distance,
    _paired_subsequence_distance,
    _paired_subsequence_match,
    _pairwise_distance,
    _pairwise_subsequence_distance,
    _singleton_pairwise_distance,
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


def _subsequence_metric_wrap(Metric):
    def f(**metric_params):
        return ScaledSubsequenceMetricWrap(Metric(**metric_params))

    return f


def _callable_metric(f):
    def wrap(**metric_params):
        return CallableMetric(f)

    return wrap


def _callable_subsequence_metric(f, scale=False):
    if scale:

        def wrap(**metric_params):
            return ScaledSubsequenceMetricWrap(CallableMetric(f))

    else:

        def wrap(**metric_params):
            return SubsequenceMetricWrap(CallableMetric(f))

    return wrap


_SUBSEQUENCE_METRICS = {
    "euclidean": EuclideanSubsequenceMetric,
    "scaled_euclidean": ScaledEuclideanSubsequenceMetric,
    "normalized_euclidean": NormalizedEuclideanSubsequenceMetric,
    "adtw": AmercingDtwSubsequenceMetric,
    "scaled_adtw": _subsequence_metric_wrap(AmercingDtwMetric),
    "wdtw": WeightedDtwSubsequenceMetric,
    "scaled_wdtw": _subsequence_metric_wrap(WeightedDtwMetric),
    "ddtw": DerivativeDtwSubsequenceMetric,
    "scaled_ddtw": _subsequence_metric_wrap(DerivativeDtwMetric),
    "wddtw": WeightedDerivativeDtwSubsequenceMetric,
    "scaled_wddtw": _subsequence_metric_wrap(WeightedDerivativeDtwMetric),
    "dtw": DtwSubsequenceMetric,
    "scaled_dtw": ScaledDtwSubsequenceMetric,
    "lcss": LcssSubsequenceMetric,
    "scaled_lcss": _subsequence_metric_wrap(LcssMetric),
    "edr": EdrSubsequenceMetric,
    "scaled_edr": _subsequence_metric_wrap(EdrMetric),
    "twe": TweSubsequenceMetric,
    "scaled_twe": _subsequence_metric_wrap(TweMetric),
    "msm": MsmSubsequenceMetric,
    "scaled_msm": _subsequence_metric_wrap(MsmMetric),
    "erp": ErpSubsequenceMetric,
    "scaled_erp": _subsequence_metric_wrap(ErpMetric),
    "mass": ScaledMassSubsequenceMetric,
    "manhattan": ManhattanSubsequenceMetric,
    "scaled_manhattan": _subsequence_metric_wrap(ManhattanMetric),
    "minkowski": MinkowskiSubsequenceMetric,
    "scaled_minkowski": _subsequence_metric_wrap(MinkowskiMetric),
    "chebyshev": ChebyshevSubsequenceMetric,
    "scaled_chebyshev": _subsequence_metric_wrap(ChebyshevMetric),
    "cosine": CosineSubsequenceMetric,
    "scaled_cosine": _subsequence_metric_wrap(CosineMetric),
    "angular": AngularSubsequenceMetric,
    "scaled_angular": _subsequence_metric_wrap(AngularMetric),
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


def _infer_scaled_metric(metric):
    if metric.startswith("scaled_"):
        return metric
    else:
        return "scaled_" + metric


def check_metric(metric):
    if callable(metric):
        return _callable_metric(metric)
    elif metric in _METRICS:
        return _METRICS[metric]
    else:
        raise ValueError(
            "unsupported metric {}, 'metric' must be callable or a str among {}".format(
                metric, set(_METRICS.keys())
            )
        )


def check_subsequence_metric(metric, scale=False):
    if callable(metric):
        return _callable_subsequence_metric(metric, scale=scale)
    else:
        if scale:
            metric = _infer_scaled_metric(metric)

        if metric in _SUBSEQUENCE_METRICS:
            return _SUBSEQUENCE_METRICS[metric]
        else:
            raise ValueError(
                (
                    "unsupported metric '{}', 'metric' "
                    "must be callable or a str among {}"
                ).format(metric, set(_SUBSEQUENCE_METRICS.keys()))
            )


def _std_below_mean(s):
    def f(x):
        return max(np.mean(x) - s * np.std(x), np.min(x))

    return f


_THRESHOLD = {"auto": _std_below_mean(2.0)}


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
    scale=False,
    return_index=False,
    n_jobs=None,
):
    """
    Minimum subsequence distance between subsequences and time series.

    Parameters
    ----------
    y : list or ndarray of shape (n_subsequences, n_timestep)
        Input time series.

        - if list, a list of array-like of shape (n_timestep, ).
    x : ndarray of shape (n_timestep, ), (n_samples, n_timestep)\
    or (n_samples, n_dims, n_timestep)
        The input data.
    dim : int, optional
        The dim to search for subsequence.
    metric : str or callable, optional
        The distance metric

        See ``_SUBSEQUENCE_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.
    scale : bool, optional
        If True, scale the subsequences before distance computation.

        .. versionadded:: 1.3
    return_index : bool, optional
        - if True return the index of the best match. If there are many equally good
          matches, the first match is returned.
    n_jobs : int, optional
        The number of parallel jobs.

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

    Warnings
    --------
    Passing a callable to the `metric` parameter has a significant performance
    implication.
    """
    y = _validate_subsequence(y)
    x = check_array(x, allow_3d=True, ensure_2d=False, dtype=np.double)
    for s in y:
        if s.shape[0] > x.shape[-1]:
            raise ValueError(
                "Invalid subsequnce shape (%d > %d)" % (s.shape[0], x.shape[-1])
            )

    Metric = check_subsequence_metric(metric, scale=scale)

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
    scale=False,
    return_index=False,
    n_jobs=None,
):
    """
    Minimum subsequence distance between the i:th subsequence and time series.

    Parameters
    ----------
    y : list or ndarray of shape (n_samples, m_timestep)
        Input time series.

        - if list, a list of array-like of shape (m_timestep, ).
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
    scale : bool, optional
        If True, scale the subsequences before distance computation.

        .. versionadded:: 1.3
    return_index : bool, optional
        - if True return the index of the best match. If there are many equally good
          matches, the first match is returned.
    n_jobs : int, optional
        The number of parallel jobs to run.

    Returns
    -------
    dist : float, ndarray
        An array of shape (n_samples, ) with the minumum distance between the i:th
        subsequence and the i:th sample.
    indices : int, ndarray, optional
        An array of shape (n_samples, ) with the index of the best matching position
        of the i:th subsequence and the i:th sample.

    Warnings
    --------
    Passing a callable to the `metric` parameter has a significant performance
    implication.
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

    Metric = check_subsequence_metric(metric, scale=scale)

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


def subsequence_match(  # noqa: PLR0912, PLR0915
    y,
    x,
    threshold=None,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    scale=False,
    max_matches=None,
    exclude=None,
    return_distance=False,
    n_jobs=None,
):
    """
    Find matching subsequnces.

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
        The subsequence.
    x : ndarray of shape (n_timestep, ), (n_samples, n_timestep)\
    or (n_samples, n_dims, n_timestep)
        The input data.
    threshold : {"auto"}, float or callable, optional
        The distance threshold used to consider a subsequence matching. If no threshold
        is selected, `max_matches` defaults to 10.

        - if float, return all matches closer than threshold
        - if callable, return all matches closer than the treshold computed by the
          threshold function, given all distances to the subsequence
        - if str, return all matches according to the named threshold.
    dim : int, optional
        The dim to search for shapelets.
    metric : str or callable, optional
        The distance metric

        See ``_SUBSEQUENCE_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.
    scale : bool, optional
        If True, scale the subsequences before distance computation.

        .. versionadded:: 1.3
    max_matches : int, optional
        Return the top `max_matches` matches below `threshold`.
    exclude : float or int, optional
        Exclude trivial matches in the vicinity of the match.

        - if float, the exclusion zone is computed as ``math.ceil(exclude * y.size)``
        - if int, the exclusion zone is exact

        A match is considered trivial if a match with lower distance is within `exclude`
        timesteps of another match with higher distance.
    return_distance : bool, optional
        - if True, return the distance of the match.
    n_jobs : int, optional
        The number of parallel jobs to run.

    Returns
    -------
    indicies : ndarray of shape (n_samples, ) or (n_matches, )
        The start index of matching subsequences. Returns a single array of
        n_matches if x.ndim == 1. If no matches are found for a sample, the
        array element is None.
    distance : ndarray of shape (n_samples, ), optional
        The distances of matching subsequences. Returns a single array of
        n_matches if x.ndim == 1. If no matches are found for a sample, the
        array element is None.

    Warnings
    --------
    Passing a callable to the `metric` parameter has a significant performance
    implication.
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

    Metric = check_subsequence_metric(metric, scale=scale)
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
        # TODO(1.3)
        if threshold == "best":
            warnings.warn(
                "threshold 'best' has been renamed to 'auto' in 1.2 "
                "and will be removed in 1.3.",
                UserWarning,
            )
            threshold = "auto"

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

    indices, distances = _subsequence_match(
        y,
        _check_ts_array(x),
        threshold,
        dim,
        Metric(**metric_params),
        n_jobs,
    )

    if max_dist is not None:
        indices, distances = _filter_by_max_dist(indices, distances, max_dist)

    if exclude:
        indices, distances = _exclude_trivial_matches(indices, distances, exclude)

    if max_matches:
        indices, distances = _filter_by_max_matches(indices, distances, max_matches)

    indices = _format_return(_safe_jagged_array(indices), len(y), x.ndim)
    if indices.size == 1:
        indices = indices.item()

    if return_distance:
        distances = _format_return(_safe_jagged_array(distances), len(y), x.ndim)
        if distances.size == 1:
            distances = distances.item()

        return indices, distances
    else:
        return indices


def paired_subsequence_match(  # noqa: PLR0912
    y,
    x,
    threshold=None,
    *,
    dim=0,
    metric="euclidean",
    metric_params=None,
    scale=False,
    max_matches=None,
    return_distance=False,
    n_jobs=None,
):
    """
    Find matching subsequnces.

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
    scale : bool, optional
        If True, scale the subsequences before distance computation.

        .. versionadded:: 1.3
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
    indicies : ndarray of shape (n_samples, )
        The start index of matching subsequences.
    distance : ndarray of shape (n_samples, ), optional
        The distances of matching subsequences.

    Warnings
    --------
    Passing a callable to the `metric` parameter has a significant performance
    implication.
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

    Metric = check_subsequence_metric(metric, scale=scale)
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

    indices, distances = _paired_subsequence_match(
        y,
        _check_ts_array(x),
        threshold,
        dim,
        Metric(**metric_params),
        n_jobs,
    )

    if max_dist is not None:
        indices, distances = _filter_by_max_dist(indices, distances, max_dist)

    if max_matches:
        indices, distances = _filter_by_max_matches(indices, distances, max_matches)

    indices = _format_return(_safe_jagged_array(indices), len(y), x.ndim)
    if indices.size == 1:
        indices = indices.reshape(1)

    if return_distance:
        distances = _format_return(_safe_jagged_array(distances), len(y), x.ndim)
        if distances.size == 1:
            distances = distances.reshape(1)

        return indices, distances
    else:
        return indices


def paired_distance(  # noqa: PLR0912
    x,
    y,
    *,
    dim="warn",
    metric="euclidean",
    metric_params=None,
    n_jobs=None,
):
    """
    Compute the distance between the i:th time series.

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

    Warnings
    --------
    Passing a callable to the `metric` parameter has a significant performance
    implication.
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

    Metric = check_metric(metric)

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


def pairwise_distance(  # noqa: PLR0912, PLR0915
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

    Warnings
    --------
    Passing a callable to the `metric` parameter has a significant performance
    implication.
    """
    Metric = check_metric(metric)
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
        "metric": [callable, StrOptions(_METRICS.keys())],
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

    Warnings
    --------
    Passing a callable to the `metric` parameter has a significant performance
    implication.

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
    metric = check_metric(metric)(**metric_params)

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


@validate_params(
    {
        "y": ["array-like"],
        "x": ["array-like"],
        "dilation": [Interval(numbers.Integral, 1, None, closed="left")],
        "padding": [
            Interval(numbers.Integral, 0, None, closed="left"),
            StrOptions({"same"}),
        ],
        "dim": [Interval(numbers.Integral, 0, None, closed="left")],
        "metric": [callable, StrOptions(_SUBSEQUENCE_METRICS.keys())],
        "metric_params": [None, dict],
        "scale": [None, bool],
        "n_jobs": [numbers.Integral, None],
    },
    prefer_skip_nested_validation=True,
)
def distance_profile(  # noqa: PLR0912
    y,
    x,
    *,
    dilation=1,
    padding=0,
    dim=0,
    metric="mass",
    metric_params=None,
    scale=False,
    n_jobs=None,
):
    """
    Compute the distance profile.

    The distance profile corresponds to the distance of the subsequences in y
    for every time point of the samples in x.

    Parameters
    ----------
    y : array-like of shape (m_timestep, ) or (n_samples, m_timestep)
        The subsequences. if `y.ndim` is 1, we will broacast `y` to have the
        same number of samples as `x`.
    x : ndarray of shape (n_timestep, ), (n_samples, n_timestep)\
    or (n_samples, n_dims, n_timestep)
        The samples. If `x.ndim` is 1, we will broadcast `x` to have the same
        number of samples as `y`.
    dilation : int, optional
        The dilation, i.e., the spacing between points in the subsequences.
    padding : int or {"same"}, optional
        The amount of padding applied to the input time series. If "same", the output
        size is the same as the input size.
    dim : int, optional
        The dim to search for shapelets.
    metric : str or callable, optional
        The distance metric

        See ``_SUBSEQUENCE_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.
    scale : bool, optional
        If True, scale the subsequences before distance computation.
    n_jobs : int, optional
        The number of parallel jobs to run.

    Returns
    -------
    ndarray of shape (n_samples, output_size) or (output_size, )
        The distance profile. `output_size` is given by:
        `n_timestep + 2 * padding - (n_timestep - 1) * dilation + 1) + 1`.
        If both `x` and `y` contains a single subsequence and a single sample,
        the output is squeezed.

    Warnings
    --------
    Passing a callable to the `metric` parameter has a significant performance
    implication.

    Examples
    --------
    >>> from wildboar.datasets import load_dataset
    >>> from wildboar.distance import distance_profile
    >>> X, _ = load_dataset("ECG200")
    >>> distance_profile(X[0], X[1:].reshape(-1))
    array([14.00120332, 14.41943788, 14.81597243, ...,  4.75219094,
           5.72681005,  6.70155561])

    >>> distance_profile(
    ...     X[0, 0:9], X[1:5], metric="dtw", dilation=2, padding="same"
    ... )[0, :10]
    array([8.01881424, 7.15083281, 7.48856368, 6.83139294, 6.75595579,
           6.30073636, 6.65346307, 6.27919601, 6.25666948, 6.0961576 ])
    """
    y = check_array(y, dtype=float, ensure_2d=False).squeeze()
    x = check_array(x, allow_3d=True, ensure_2d=False, dtype=float).squeeze()

    if x.ndim == 1 and y.ndim != 1:
        x = np.broadcast_to(x, shape=(y.shape[0], x.shape[0]))

    if y.ndim == 1 and x.ndim != 1:
        y = np.broadcast_to(y, shape=(x.shape[0], y.shape[0]))

    x = _check_ts_array(x)
    y = _check_ts_array(y)

    shapelet_size = (y.shape[2] - 1) * dilation + 1
    if padding == "same":
        if y.shape[2] % 2 == 0:
            raise ValueError(
                "padding='same' is only supported for odd subsequence length"
            )
        padding = shapelet_size // 2

    input_size = x.shape[2] + 2 * padding

    if shapelet_size > input_size:
        raise ValueError("subsequence in y is larger than input in x.")

    if y.shape[0] != x.shape[0]:
        raise ValueError("y and x must have the same number of samples.")

    if dim >= x.shape[1]:
        raise ValueError(
            f"The parameter dim must be dim ({dim}) < n_dims ({x.shape[1]})"
        )

    if dilation == 1 and padding == 0:
        Metric = check_subsequence_metric(metric, scale=scale)
        metric_params = metric_params if metric_params is not None else {}
        dp = _distance_profile(y, x, dim, Metric(**metric_params), n_jobs)
    else:
        # HACK
        if metric == "mass":
            metric = "scaled_euclidean"

        scale = (isinstance(metric, str) and metric.startswith("scaled_")) or scale
        if isinstance(metric, str) and metric.startswith("scaled_"):
            metric = metric[7:]

        Metric = check_metric(metric)
        metric_params = metric_params if metric_params is not None else {}

        if scale:
            std = np.std(y, axis=-1, keepdims=True)
            mean = np.mean(y, axis=-1, keepdims=True)
            std[std < 1e-13] = 1  # NOTE! see EPSILON in _cdistance.pxd
            y = (y - mean) / std

        dp = _dilated_distance_profile(
            y,
            x,
            dim,
            Metric(**metric_params),
            dilation,
            padding,
            scale,
            n_jobs,
        )

    return np.squeeze(dp)


@validate_params(
    {
        "y": ["array-like"],
        "x": ["array-like"],
        "dim": [Interval(numbers.Integral, 0, None, closed="left")],
        "k": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric": [callable, StrOptions(_SUBSEQUENCE_METRICS.keys() - {"mass"})],
        "metric_params": [None, dict],
        "scale": [None, bool],
        "return_distance": [bool],
        "n_jobs": [numbers.Integral, None],
    },
    prefer_skip_nested_validation=True,
)
def argmin_subsequence_distance(  # noqa: PLR0912
    y,
    x,
    *,
    dim=0,
    k=1,
    metric="euclidean",
    metric_params=None,
    scale=False,
    return_distance=False,
    n_jobs=None,
):
    """
    Compute the k:th closest subsequences.

    For the i:th shapelet and the i:th sample return the index and, optionally,
    the distance of the `k` closest matches.

    Parameters
    ----------
    y : array-like of shape (n_samples, m_timestep) or list of 1d-arrays
        The subsequences.
    x : array-like of shape (n_timestep, ), (n_samples, n_timestep)\
    or (n_samples, n_dims, n_timestep)
        The samples. If x.ndim == 1, it will be broadcast have the same number
        of samples that y.
    dim : int, optional
        The dimension in x to find subsequences in.
    k : int, optional
        The of closest subsequences to find.
    metric : str, optional
        The metric.

        See ``_SUBSEQUENCE_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_subsequence_metrics>`.
    scale : bool, optional
        If True, scale the subsequences before distance computation.
    return_distance : bool, optional
        Return the distance for the `k` closest subsequences.
    n_jobs : int, optional
       The number of parallel jobs.

    Returns
    -------
    indices : ndarray of shape (n_samples, k)
        The indices of the `k` closest subsequences.
    distance : ndarray of shape (n_samples, k), optional
        The distance of the `k` closest subsequences.

    Warnings
    --------
    Passing a callable to the `metric` parameter has a significant performance
    implication.

    Examples
    --------
    >>> import numpy as np
    >>> from wildboar.datasets import load_dataset
    >>> from wildboar.distance import argmin_subsequence_distance
    >>> s = np.lib.stride_tricks.sliding_window_view(X[0], window_shape=10)
    >>> x = np.broadcast_to(X[0], shape=(s.shape[0], X.shape[1]))
    >>> argmin_subsequence_distance(s, x, k=4)
    """
    if isinstance(y, np.ndarray) and y.dtype == float:
        y = check_array(y, allow_3d=False, ensure_ts_array=True)
        y_len = np.broadcast_to(y.shape[2], y.shape[0]).astype(np.intp)
    else:
        # we assume y is a list of ndarrays
        y_len = np.empty(len(y), dtype=np.intp)
        for i, shapelet in enumerate(y):
            if shapelet.ndim > 1:
                raise ValueError("shapelet must be 1d-array")

            y_len[i] = shapelet.shape[0]

        y_tmp = np.empty((len(y), 1, np.max(y_len)), dtype=float)
        for i, shapelet in enumerate(y):
            y_tmp[i, 0, : shapelet.shape[0]] = shapelet

        y = y_tmp

    if x.ndim == 1:
        x = np.broadcast_to(x, shape=(y.shape[0], x.shape[0]))

    x = check_array(x, allow_3d=True, ensure_ts_array=True)

    if not y.shape[2] <= x.shape[2]:
        raise ValueError("the longest subsequence must be shorter than samples.")

    if y.shape[0] != x.shape[0]:
        raise ValueError("both arrays must have the same number of samples.")

    if dim >= x.shape[1]:
        raise ValueError(
            f"The parameter dim must be dim ({dim}) < n_dims ({x.shape[1]})"
        )

    if not k <= (x.shape[2] - y.shape[2] + 1):
        raise ValueError("k must be less x.shape[-1] - y.shape[-1] + 1.")

    scale = (isinstance(metric, str) and metric.startswith("scaled_")) or scale
    if isinstance(metric, str) and metric.startswith("scaled_"):
        metric = metric[7:]

    Metric = check_metric(metric)
    metric_params = metric_params if metric_params is not None else {}

    indices, distances = _argmin_subsequence_distance(
        y,
        y_len,
        x,
        dim,
        Metric(**metric_params),
        k,
        scaled=bool(scale),
        n_jobs=n_jobs,
    )

    if return_distance:
        return indices, distances
    else:
        return indices
