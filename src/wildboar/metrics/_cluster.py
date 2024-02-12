import numbers

from sklearn.metrics import silhouette_samples as sklearn_silhouette_samples
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
    validate_params,
)
from sklearn.utils.validation import check_random_state

from ..distance._distance import _METRICS, pairwise_distance


@validate_params(
    {
        "X": ["array-like"],
        "labels": ["array-like"],
        "metric": [StrOptions(_METRICS.keys()), callable],
        "metric_params": [dict, None],
        "sample_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def silhouette_score(
    x,
    labels,
    *,
    metric="euclidean",
    metric_params=None,
    sample_size=None,
    random_state=None,
):
    """
    Compute the mean Silhouette Coefficient of all samples.

    Parameters
    ----------
    x : univariate time-series or multivariate time-series
        The input time series.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    metric : str or callable, optional
        The metric to use when calculating distance between time series.
    metric_params : dict, optional
        The metric parameters. Read more about the metrics and their parameters
        in the :ref:`User guide <list_of_metrics>`.
    sample_size : int, optional
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    random_state : int or RandomState, optional
        Determines random number generation for selecting a subset of samples.
        Used when ``sample_size is not None``.

    Returns
    -------
    float
        Mean Silhouette Coefficient for all samples.

    Notes
    -----
    This is a convenient wrapper around :ref:`sklearn.metrics.silhouette_score`
    using Wildboar native metrics.
    """
    if sample_size is not None:
        random_state = check_random_state(random_state)
        idx = random_state.choice(x.shape[0], sample_size, replace=False)
        x = x[idx]
        labels = labels[idx]

    return sklearn_silhouette_score(
        pairwise_distance(x, metric=metric, metric_params=metric_params, dim="mean"),
        labels,
        metric="precomputed",
        sample_size=None,
        random_state=None,
    )


@validate_params(
    {
        "X": ["array-like"],
        "labels": ["array-like"],
        "metric": [StrOptions(_METRICS.keys()), callable],
        "metric_params": [dict, None],
    },
    prefer_skip_nested_validation=True,
)
def silhouette_samples(x, labels, *, metric="euclidean", metric_params=None):
    """
    Compute the Silhouette Coefficient of each samples.

    Parameters
    ----------
    x : univariate time-series or multivariate time-series
        The input time series.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    metric : str or callable, optional
        The metric to use when calculating distance between time series.
    metric_params : dict, optional
        The metric parameters. Read more about the metrics and their parameters
        in the :ref:`User guide <list_of_metrics>`.

    Returns
    -------
    ndarray of shape (n_samples, )
        Silhouette Coefficient for each samples.

    Notes
    -----
    This is a convenient wrapper around :ref:`sklearn.metrics.silhouette_samples`
    using Wildboar native metrics.
    """
    return sklearn_silhouette_samples(
        pairwise_distance(x, metric=metric, metric_params=metric_params, dim="mean"),
        labels,
        metric="precomputed",
    )
