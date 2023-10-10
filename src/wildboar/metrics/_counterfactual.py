import math
import numbers

import numpy as np
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor

from ..distance import paired_distance, pairwise_distance
from ..explain._importance import _intervals
from ..explain.counterfactual import proximity
from ..transform._sax import (
    piecewice_aggregate_approximation,
    symbolic_aggregate_approximation,
)
from ..utils.validation import check_array


def _estimate_plausability(estimator, X_counterfactuals, method):
    if method == "score":
        return np.mean(estimator.decision_function(X_counterfactuals))
    elif method == "accuracy":
        y_true = np.broadcast_to(1, X_counterfactuals.shape[0])
        return accuracy_score(y_true, estimator.predict(X_counterfactuals))
    else:
        raise ValueError("method must be 'average', or 'accuracy', " "got %r" % method)


def plausability_score(
    x_plausible,
    x_counterfactuals,
    *,
    y_plausible=None,
    y_counterfactual=None,
    estimator=None,
    method="accuracy",
    average=True,
):
    """
    Compute plausibility score.

    Parameters
    ----------
    x_plausible : array-like of shape (n_samples, n_timesteps)
        The plausible samples, typically the training or testing samples.
    x_counterfactuals : array-like of shape (m_samples, n_timesteps)
        The counterfactual samples.
    y_plausible : array-like of shape (n_samples, ), optional
        The labels of the plausible samples.
    y_counterfactual : array-like of shape (m_samples, ), optional
        The desired label of the counterfactuals.
    estimator : estimator, optional
        The outlier estimator, must implement `fit` and `predict`. If None,
        we use LocalOutlierFactor.

        - if score="mean", the estimator must also implement `decision_function`.
    method : {'score', 'accuracy'}, optional
        The score function.
    average : bool, optional
        If True, return the average score for all labels in y_counterfactual;
        otherwise, return the score for the individual labels (ordered as np.unique).

    Returns
    -------
    ndarray or float
        The plausability.

        - if method='scores', the mean score is returned, with larger score incicating
          better performance.
        - if method='accuracy', the fraction of plausible counterfactuals are returned.
        - if y_counterfactual is None and average=False, the scores or accuracy for each
          counterfactual label is returned.

    References
    ----------
    Delaney, E., Greene, D., & Keane, M. T. (2020).
        Instance-based Counterfactual Explanations for Time Series Classification.
        arXiv, 2009.13211v2.
    """
    if estimator is None:
        estimator = LocalOutlierFactor(
            n_neighbors=math.ceil(np.sqrt(x_plausible.shape[0])), novelty=True
        )
    else:
        estimator = clone(estimator)

    x_plausible = check_array(x_plausible, allow_3d=True)
    x_counterfactuals = check_array(x_counterfactuals, allow_3d=True)
    if x_plausible.shape[-1] != x_counterfactuals.shape[-1]:
        raise ValueError(
            "X_plausible (%s) and X_counterfactuals (%s) must have the same number "
            "of timesteps." % (x_plausible.shape, x_counterfactuals.shape)
        )
    if y_counterfactual is None:
        estimator.fit(x_plausible)
        return _estimate_plausability(estimator, x_counterfactuals, method)
    else:
        if y_plausible is None:
            raise ValueError(
                "if y_counterfactual is given, y_plausible must also be given"
            )

        y_plausible = check_array(
            y_plausible, ensure_2d=False, ravel_1d=True, dtype=None
        )
        y_counterfactual = check_array(
            y_counterfactual, ensure_2d=False, ravel_1d=True, dtype=None
        )

        labels = np.unique(y_counterfactual)
        scores = []
        for label in labels:
            X_plausible_label = x_plausible[y_plausible == label]
            if X_plausible_label.shape[0] == 0:
                raise ValueError(f"Not enough plausable samples with label={label}.")

            label_estimator = clone(estimator)
            label_estimator.fit(X_plausible_label)
            scores.append(
                _estimate_plausability(
                    label_estimator,
                    x_counterfactuals[y_counterfactual == label],
                    method,
                )
            )

        scores = np.array(scores, dtype=float)
        return scores.mean() if average else scores


def relative_proximity_score(
    x_native,
    x_factual,
    x_counterfactual,
    *,
    y_native=None,
    y_counterfactual=None,
    metric="euclidean",
    metric_params=None,
    average=True,
):
    """
    Compute relative proximity score.

    The relative proximity score captures the mean proximity of counterfactual
    and test sample pairs over mean proximity of the closest native
    counterfactual. The lower the score, the better.

    Parameters
    ----------
    x_native : array-like of shape (n_natives, n_timesteps)
        The native counterfactual candidates. If y_counterfactual is None, the full
        array is considered as possible native counterfactuals. Typically, native
        counterfactual candidates correspond to samples which are labeled as the
        desired counterfactual label.
    x_factual : array-like of shape (n_counterfactuals, n_timesteps)
        The factual samples, i.e., the samples for which the counterfactuals
        where computed.
    x_counterfactual : array-like of shape (n_counterfactuals, n_timesteps)
        The counterfactual samples.
    y_native : array-like of shape (n_natives, ), optional
        The label of the native counterfactual candidates.
    y_counterfactual : array-like of shape (n_counterfactuals, ), optional
        The desired counterfactual label.
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.
    average : bool, optional
        Average the relative proximity of all labels in y_counterfactual.

    Returns
    -------
    ndarray or float
        The relative proximity. If avarege=False and y_counterfactual is not None,
        return the relative proximity for each counterfactual label.

    Notes
    -----
    The samples in `x_counterfactual` and `x_factual` should be aligned such
    that the i:th counterfacutal sample is derived from the i:th factual sample.

    References
    ----------
    Smyth, B., & Keane, M. T. (2021).
        A Few Good Counterfactuals: Generating Interpretable, Plausible and Diverse
        Counterfactual Explanations. arXiv, 2101.09056v1.
    """
    x_native = check_array(x_native)
    x_factual = check_array(x_factual)
    x_counterfactual = check_array(x_counterfactual)

    cf_dist = paired_distance(
        x_factual, x_counterfactual, metric=metric, metric_params=metric_params
    ).mean()
    if y_counterfactual is None:
        native_dist = pairwise_distance(
            x_native, x_factual, metric=metric, metric_params=metric_params
        )
        return cf_dist / native_dist.min(axis=0).mean()
    else:
        if y_native is None:
            raise ValueError("if y_counterfactual is give, y_native must also be given")

        y_native = check_array(y_native, ensure_2d=False, ravel_1d=True, dtype=None)
        y_counterfactual = check_array(
            y_counterfactual, ensure_2d=False, ravel_1d=True, dtype=None
        )
        cf_labels = np.unique(y_counterfactual)
        native_dists = []
        for label in cf_labels:
            X_native_cf_label = x_native[y_native == label]
            if X_native_cf_label.shape[0] == 0:
                raise ValueError(f"Not enough native samples with label={label}.")

            native_dist = pairwise_distance(
                X_native_cf_label,
                x_factual[y_counterfactual == label],
                metric=metric,
                metric_params=metric_params,
            )
            native_dists.append(native_dist.min(axis=0).mean())

        native_dists = np.array(native_dists, dtype=float)
        return cf_dist / (native_dists.mean() if average else native_dists)


def proximity_score(
    x_factual,
    x_counterfactual,
    metric="normalized_euclidean",
    metric_params=None,
):
    """
    Compute proximity score.

    The closer the counterfactual is to the original, the lower the score.

    Parameters
    ----------
    x_factual : array-like of shape (n_samples, n_timestep)
        The true samples.
    x_counterfactual : array-like of shape (n_samples, n_timestep)
        The counterfactual samples.
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.

    Returns
    -------
    float
        The mean proximity.

    Notes
    -----
    The samples in `x_counterfactual` and `x_factual` should be aligned such
    that the i:th counterfacutal sample is derived from the i:th factual sample.

    References
    ----------
    Delaney, E., Greene, D., & Keane, M. T. (2020).
        Instance-based Counterfactual Explanations for Time Series Classification.
        arXiv, 2009.13211v2.
    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2020).
        Locally and globally explainable time series tweaking.
        Knowledge and Information Systems, 62(5), 1671-1700.
    """
    x_true = check_array(x_factual, allow_3d=True)
    x_counterfactuals = check_array(x_counterfactual, allow_3d=True)

    return np.mean(
        proximity(x_true, x_counterfactuals, metric=metric, metric_params=metric_params)
    )


def compactness_score(
    x_factual,
    x_counterfactual,
    *,
    window=None,
    n_bins=None,
    atol=1.0e-8,
    average=True,
):
    """
    Compute compactness score.

    The compactness of counterfactuals as measured by the fraction of changed
    timesteps. The fewer timesteps have changed between the original and the
    counterfactual, the lower the score.

    Parameters
    ----------
    x_factual : array-like of shape (n_samples, n_timesteps) \
            or (n_samples, n_dims, n_timeteps)
        The true samples.
    x_counterfactual : array-like of shape (n_samples, n_timesteps) \
            or (n_samples, n_dims, n_timeteps)
        The counterfactual samples.
    window : int, optional
        If set, evaluate the difference between windows of specified size.
    n_bins : int, optional
        If set, evaluate the set overlap of SAX transformed series.
    atol : float, optional
        The absolute tolerance.
    average : bool, optional
        Compute average score over all dimensions.

    Returns
    -------
    float
        The compactness score. Lower score indicates more compact counterfactuals.

    Notes
    -----
    The samples in `x_counterfactual` and `x_factual` should be aligned such
    that the i:th counterfacutal sample is derived from the i:th factual sample.

    References
    ----------
    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2020).
        Locally and globally explainable time series tweaking.
        Knowledge and Information Systems, 62(5), 1671-1700.
    """
    x_factual = check_array(x_factual, allow_3d=True)
    x_counterfactual = check_array(x_counterfactual, allow_3d=True)
    if x_factual.shape != x_counterfactual.shape:
        raise ValueError(
            "x_true (%s) and x_counterfactuals (%s) must have the same shape."
            % (x_factual.shape, x_counterfactual.shape)
        )

    if window is not None:
        if n_bins is not None:

            def score(x_counterfactual, x_factual):
                x_counterfactual = symbolic_aggregate_approximation(
                    x_counterfactual, window=window, n_bins=n_bins
                )
                x_factual = symbolic_aggregate_approximation(
                    x_factual, window=window, n_bins=n_bins
                )
                return x_counterfactual == x_factual

        else:

            def score(x_counterfactual, x_factual):
                x_counterfactual = piecewice_aggregate_approximation(
                    x_counterfactual, window=window
                )
                x_factual = piecewice_aggregate_approximation(x_factual, window=window)
                return np.isclose(x_counterfactual, x_factual, rtol=0, atol=atol)

    else:

        def score(x_counterfactual, x_true):
            return np.isclose(x_counterfactual, x_true, rtol=0, atol=atol)

    return 1 - np.mean(score(x_counterfactual, x_factual), axis=None if average else 0)


def validity_score(y_predicted, y_counterfactual, sample_weight=None):
    """
    Compute validity score.

    The number counterfactuals that have the desired label.

    Parameters
    ----------
    y_predicted : array-like of shape (n_samples, )
        The predicted label.
    y_counterfactual : array-like of shape (n_samples, )
        The predicted label.
    sample_weight : array-like of shape (n_samples, ), optional
        The sample weight.

    Returns
    -------
    float
        The fraction of counterfactuals with the correct label. Larger is better.

    References
    ----------
    Delaney, E., Greene, D., & Keane, M. T. (2020).
        Instance-based Counterfactual Explanations for Time Series Classification.
        arXiv, 2009.13211v2.
    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2020).
        Locally and globally explainable time series tweaking.
        Knowledge and Information Systems, 62(5), 1671-1700.
    """
    return accuracy_score(y_predicted, y_counterfactual, sample_weight=sample_weight)


def redudancy_score(
    estimator,
    x_factual,
    x_counterfactual,
    y_counterfactual,
    *,
    n_intervals="sqrt",
    window=None,
    average=True,
):
    """
    Compute the redudancy score.

    Redundancy is measure of how much impact non-overlapping intervals has
    in the construction of the counterfactuals.

    Parameters
    ----------
    estimator : Estimator
        The estimator counterfactuals are computed for.
    x_factual : array-like of shape (n_samples, n_timestep)
        The factual samples, i.e., samples for which counterfactuals
        are computed.
    x_counterfactual : array-like of shape (n_samples, n_timestep)
        The counterfactual samples.
    y_counterfactual : array-like of shape (n_samples, )
        The desired counterfactual label.
    n_intervals : {"sqrt", "log2"}, int or float, optional
        The number of intervals.
    window : int, optional
        The size of an interval. If set, `n_intervals` is ignored.
    average : bool, optional
        Return the average redundancy over all intervals.

    Returns
    -------
    ndarray of shape (n_intervals, ) or float
        The redundancy of each interval, expressed as the fraction
        of samples that have the same label if the interval is replaced
        with the corresponding interval of the factual sample. If `average`
        is True, return a single float.

    Notes
    -----
    The samples in `x_counterfactual` and `x_factual` should be aligned such
    that the i:th counterfacutal sample is derived from the i:th factual sample.
    """
    if window is not None:
        if window > x_counterfactual.shape[-1]:
            raise ValueError(
                "The window parameter must be <= X_counterfactual.shape[-1]"
            )
        n_intervals = x_counterfactual.shape[-1] // window
    elif n_intervals == "sqrt":
        n_intervals = math.ceil(math.sqrt(x_counterfactual.shape[-1]))
    elif n_intervals in {"log", "log2"}:
        n_intervals = math.ceil(math.log2(x_counterfactual.shape[-1]))
    elif isinstance(n_intervals, numbers.Integral):
        if n_intervals > x_counterfactual.shape[-1]:
            raise ValueError(
                "The n_intervals parameter must be <= X_counterfactual.shape[-1]"
            )
    else:
        n_intervals = math.ceil(x_counterfactual.shape[-1] * n_intervals)

    x_counterfactual = check_array(x_counterfactual)
    x_actual = check_array(x_factual)
    y_counterfactual = check_array(
        y_counterfactual, dtype=None, ravel_1d=True, ensure_2d=False
    )

    if x_counterfactual.shape != x_actual.shape:
        raise ValueError(
            f"x_factual ({x_factual.shape}) and x_counterfactuals ({x_factual.shape}) "
            "must have the same shape."
        )

    if x_counterfactual.shape[0] != y_counterfactual.shape[0]:
        raise ValueError(
            "x_counterfactual and y_factual must have the same number of samples "
            f"{x_counterfactual.shape[0]} != {y_counterfactual.shape[0]}"
        )

    r = np.zeros(n_intervals, dtype=float)

    # indicator array of timesteps that have been changed.
    mask = ~np.isclose(x_counterfactual, x_actual)
    for i, (start, end) in enumerate(
        _intervals(x_counterfactual.shape[-1], n_intervals)
    ):
        # samples where there is a difference between the counterfactual
        # sample and the factual sample. i.e., the interval has been changed.
        idx = np.any(mask[:, start:end], axis=1)
        if np.any(idx):  # TODO: should this be all?
            x_tmp = x_counterfactual[idx, :].copy()
            x_tmp[:, start:end] = x_actual[idx, start:end]
            r[i] = (estimator.predict(x_tmp) == y_counterfactual[idx]).sum() / idx.sum()

    return r.mean() if average else r
