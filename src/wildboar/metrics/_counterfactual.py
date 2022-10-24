import numpy as np
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor

from ..explain.counterfactual import proximity
from ..utils.validation import check_array


def plausability_score(X_plausible, X_counterfactuals, estimator=None):
    """Compute the plausibility of the generated counterfactuals.

    Parameters
    ----------
    X_plausible : array-like of shape (n_samples, n_timesteps)
        The plausible time series, typically the training or testing samples.

    X_counterfactuals : array-like of shape (m_samples, n_timesteps)
        The counterfactuals generated

    estimator : Estimator, optional
        The outlier estimator.

    Returns
    -------
    score : float
        The mean plausability. Larger score is better, negative scores indicate many
        outliers.
    """
    if estimator is None:
        estimator = LocalOutlierFactor(n_neighbors=1, novelty=True)
    else:
        estimator = clone(estimator)

    X_plausible = check_array(X_plausible, allow_3d=True)
    X_counterfactuals = check_array(X_counterfactuals, allow_3d=True)
    return np.mean(estimator.fit(X_plausible).decision_function(X_counterfactuals))


def proximity_score(
    x_true,
    x_counterfactuals,
    normalize=False,
    kernel_width=None,
    metric="euclidean",
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

    normalize : bool, optional
        Normalize the score in [0, 1], with 1 indicating perfect proximity.

    metric : str or callable, optional
        The scoring metric

        - if str use metrics from scikit-learn or wildboar

    Returns
    -------
    score : float
        The mean proximity.

        - if normalize=True, higher score is better.
        - if normalize=False, lower score is better.
    """
    x_true = check_array(x_true, allow_3d=True)
    x_counterfactuals = check_array(x_counterfactuals, allow_3d=True)

    return np.mean(
        proximity(
            x_true,
            x_counterfactuals,
            normalize=normalize,
            kernel_width=kernel_width,
            metric=metric,
            metric_params=metric_params,
        )
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
    """
    return accuracy_score(y_pred, y_counterfactual, sample_weight=sample_weight)
