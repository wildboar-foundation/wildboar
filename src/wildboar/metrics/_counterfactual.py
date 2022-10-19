import numpy as np
from sklearn.metrics import accuracy_score

from ..distance import mean_paired_distance
from ..utils.validation import check_array


def proximity_score(x_true, x_counterfactuals, metric="euclidean", metric_params=None):
    """Compute the proximity score of the counterfactuals using the provided metric.

    The closer the counterfactual is to the original, the lower the score.

    Parameters
    ----------
    x_true : array-like of shape (n_samples, n_timestep)
        The true samples

    x_counterfactuals : array-like of shape (n_samples, n_timestep)
        The counterfactual samples

    metric : str or callable, optional
        The scoring metric

        - if str use metrics from scikit-learn or wildboar

    Returns
    -------
    score : ndarray or dict
        The scores
    """
    x_true = check_array(x_true, allow_3d=True)
    x_counterfactuals = check_array(x_counterfactuals, allow_3d=True)

    if isinstance(metric, str) or callable(metric):
        return np.mean(
            mean_paired_distance(
                x_true, x_counterfactuals, metric=metric, metric_params=metric_params
            )
        )
    else:
        raise TypeError(
            "metric should be str or callable, not %r" % type(metric).__qualname__
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
    """
    x_true = check_array(x_true, allow_3d=True)
    x_counterfactuals = check_array(x_counterfactuals, allow_3d=True)
    if x_true.shape != x_counterfactuals.shape:
        raise ValueError(
            "x_true (%s) and x_counterfactuals (%s) must have the same shape."
            % (x_true.shape, x_counterfactuals.shape)
        )

    return 1 - np.mean(np.isclose(x_counterfactuals, x_true, rtol=rtol, atol=atol))


def validity_score(y_counterfactual, y_pred):
    return accuracy_score(y_counterfactual, y_pred)
