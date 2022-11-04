import math

import numpy as np
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor

from ..distance import paired_distance, pairwise_distance
from ..explain.counterfactual import proximity
from ..utils.validation import check_array


def plausability_score(
    X_plausible, X_counterfactuals, estimator=None, method="accuracy"
):
    """Compute the plausibility of the generated counterfactuals.

    Parameters
    ----------
    X_plausible : array-like of shape (n_samples, n_timesteps)
        The plausible time series, typically the training or testing samples.

    X_counterfactuals : array-like of shape (m_samples, n_timesteps)
        The counterfactuals generated

    estimator : Estimator, optional
        The outlier estimator.

    method : {'average', 'accuracy'}, optional
        The score function.

    Returns
    -------
    score : float
        The plausability.

        - if method='average', the mean score is returned, with larger score incicating
          better performance.

        - if method='accuracy', the fraction of plausible counterfactuals are returned.

    References
    ----------
    Delaney, E., Greene, D., & Keane, M. T. (2020).
        Instance-based Counterfactual Explanations for Time Series Classification.
        arXiv, 2009.13211v2.
    """
    if estimator is None:
        estimator = LocalOutlierFactor(
            n_neighbors=math.ceil(np.sqrt(X_plausible.shape[0])), novelty=True
        )
    else:
        estimator = clone(estimator)

    X_plausible = check_array(X_plausible, allow_3d=True)
    X_counterfactuals = check_array(X_counterfactuals, allow_3d=True)
    if X_plausible.shape[-1] != X_counterfactuals.shape[-1]:
        raise ValueError(
            "X_plausible (%s) and X_counterfactuals (%s) must have the same number "
            "of timesteps." % (X_plausible.shape, X_counterfactuals.shape)
        )
    estimator.fit(X_plausible)
    y_true = np.broadcast_to(1, X_counterfactuals.shape[0])
    if method == "average":
        return np.mean(estimator.decision_function(X_counterfactuals))
    elif method == "accuracy":
        return accuracy_score(y_true, estimator.predict(X_counterfactuals))
    else:
        raise ValueError("method must be 'average', or 'accuracy', " "got %r" % method)


def relative_proximity_score(
    X_native,
    X_test,
    X_counterfactual,
    *,
    y_native=None,
    y_counterfactual=None,
    metric="euclidean",
    metric_params=None,
):
    """Relative proximity score captures the mean proximity of counterfactual and
    test sample pairs over mean proximity of the closest native counterfactual. The
    lower the score, the better.

    Parameters
    ----------
    X_native : array-like of shape (n_natives, n_timesteps)
        The native counterfactual candidates. If y_counterfactual is None, the full
        array is considered as possible native counterfactuals.

    X_test : array-like of shape (n_counterfactuals, n_timesteps)
        The test samples.

    X_counterfactuals : array-like of shape (n_counterfactuals, n_timesteps)
        The counterfactual samples.

    y_native : array-like of shape (n_natives, ), optional
        The label of the native counterfactual candidates.

    y_counterfactual : array-like of shape (n_counterfactuals, )
        The desired counterfactual label.

    metric : str or callable, optional
        The distance metric

        See ``_DISTANCE_MEASURE.keys()`` for a list of supported metrics.

    metric_params: dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.

    Returns
    -------
    score : float
        The relative proximity.

    References
    ----------
    Smyth, B., & Keane, M. T. (2021).
        A Few Good Counterfactuals: Generating Interpretable, Plausible and Diverse
        Counterfactual Explanations. arXiv, 2101.09056v1.
    """
    X_native = check_array(X_native)
    X_test = check_array(X_test)
    X_counterfactual = check_array(X_counterfactual)

    cf_dist = paired_distance(
        X_test, X_counterfactual, metric=metric, metric_params=metric_params
    ).mean()
    if y_counterfactual is None:
        native_dist = pairwise_distance(
            X_native, X_test, metric=metric, metric_params=metric_params
        )
        return cf_dist / native_dist.min(axis=0).mean()
    else:
        if y_native is None:
            raise ValueError("if y_counterfactual is give, y_native must also be given")

        y_native = check_array(y_native, dtype=None)
        y_counterfactual = check_array(y_counterfactual, dtype=None)
        cf_labels = np.unique(y_counterfactual)
        native_dist_sum = 0
        for label in cf_labels:
            X_native_cf_label = X_native[y_native == label]
            if X_native_cf_label.shape[0] == 0:
                raise ValueError(f"Not enough native samples with label={label}.")

            native_dist = pairwise_distance(
                X_native_cf_label,
                X_test[y_counterfactual == label],
                metric=metric,
                metric_params=metric_params,
            )
            native_dist_sum += native_dist.min(axis=0).mean()

        return cf_dist / (native_dist_sum / len(cf_labels))


def proximity_score(
    x_true,
    x_counterfactuals,
    metric="normalized_euclidean",
    metric_params=None,
):
    """Compute the proximity score of the counterfactuals using the provided metric.

    The closer the counterfactual is to the original, the lower the score.

    Parameters
    ----------
    x_true : array-like of shape (n_samples, n_timestep)
        The true samples

    x_counterfactuals : array-like of shape (n_samples, n_timestep)
        The counterfactual samples

    metric : str or callable, optional
        The distance metric

        See ``_DISTANCE_MEASURE.keys()`` for a list of supported metrics.

    metric_params: dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.

    Returns
    -------
    score : float
        The mean proximity.

    References
    ----------
    Delaney, E., Greene, D., & Keane, M. T. (2020).
        Instance-based Counterfactual Explanations for Time Series Classification.
        arXiv, 2009.13211v2.

    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2020).
        Locally and globally explainable time series tweaking.
        Knowledge and Information Systems, 62(5), 1671-1700.
    """
    x_true = check_array(x_true, allow_3d=True)
    x_counterfactuals = check_array(x_counterfactuals, allow_3d=True)

    return np.mean(
        proximity(x_true, x_counterfactuals, metric=metric, metric_params=metric_params)
    )


def compactness_score(x_true, x_counterfactuals, *, rtol=1.0e-5, atol=1.0e-8):
    """Return the compactness of the counterfactuals as measured by the
    fraction of changed timesteps. The fewer timesteps have changed between the original
    and the counterfactual, the lower the score.

    Parameters
    ----------
    x_true : array-like of shape (n_samples, n_timesteps) \
    or (n_samples, n_dims, n_timeteps)
        The true samples

    x_counterfactuals : array-like of shape (n_samples, n_timesteps) \
    or (n_samples, n_dims, n_timeteps)
        The counterfactual samples

    rtol : float, optional
        Parameter to `np.isclose`.

    atol : float, optional
        Parameter to `np.isclose`.

    Returns
    -------
    score : float
        The compactness score. Lower score indicates more compact counterfactuals.

    References
    ----------
    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2020).
        Locally and globally explainable time series tweaking.
        Knowledge and Information Systems, 62(5), 1671-1700.
    """
    x_true = check_array(x_true, allow_3d=True)
    x_counterfactuals = check_array(x_counterfactuals, allow_3d=True)
    if x_true.shape != x_counterfactuals.shape:
        raise ValueError(
            "x_true (%s) and x_counterfactuals (%s) must have the same shape."
            % (x_true.shape, x_counterfactuals.shape)
        )

    return 1 - np.mean(np.isclose(x_counterfactuals, x_true, rtol=rtol, atol=atol))


def validity_score(y_pred, y_counterfactual, sample_weight=None):
    """Compute the number counterfactuals that have the desired label.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples, )
        The desired label
    y_counterfactual : array-like of shape (n_samples, )
        The predicted label
    sample_weight : array-like of shape (n_samples, ), optional
        The sample weight

    Returns
    -------
    score : float
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
    return accuracy_score(y_pred, y_counterfactual, sample_weight=sample_weight)
