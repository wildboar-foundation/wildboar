# Authors: Isak Samsten
# License: BSD 3 clause

import math

import numpy as np
from sklearn import model_selection
from sklearn.utils import check_random_state
from sklearn.utils._random import sample_without_replacement

from ..model_selection._cv import RepeatedOutlierSplit

__all__ = [
    "train_test_split",
    "threshold_score",
    "RepeatedOutlierSplit",
]


def train_test_split(
    x, y, normal_class, test_size=0.2, anomalies_train_size=0.05, random_state=None
):
    """Training and testing split from classification dataset

    Parameters
    ----------
    x : array-like of shape (n_samples, n_timestep) or (n_samples, n_dim, n_timestep)
        Input data samples

    y : array-like of shape (n_samples,)
        Input class label

    normal_class : int
        Class label that should be considered as the normal class

    test_size : float
        Size of the test set

    anomalies_train_size : float
        Contamination of anomalies in the training dataset

    random_state : int or RandomState
        Psudo random state used for stable results.

    Returns
    -------
    x_train : array-like
        Training samples

    x_test : array-like
        Test samples

    y_train : array-like
        Training labels (either 1 or -1, where 1 denotes normal and -1 anomalous)

    y_test : array-like
        Test labels (either 1 or -1, where 1 denotes normal and -1 anomalous)

    Examples
    --------

    >>> from wildboar.datasets import load_two_lead_ecg
    >>> x, y = load_two_lead_ecg()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     x, y, 1, test_size=0.2, anomalies_train_size=0.05
    ... )
    """
    random_state = check_random_state(random_state)
    normal = y == normal_class
    y = y.copy()
    y[normal] = 1
    y[~normal] = -1

    x_normal = x[np.where(y == 1)]
    x_anomalous = x[np.where(y == -1)]
    y_normal = y[np.where(y == 1)]
    y_anomalous = y[np.where(y == -1)]

    (xn_train, xn_test, yn_train, yn_test) = model_selection.train_test_split(
        x_normal, y_normal, test_size=test_size, random_state=random_state
    )

    n_sample = min(x_anomalous.shape[0], x_normal.shape[0])
    idx = sample_without_replacement(
        x_anomalous.shape[0], n_sample, random_state=random_state
    )
    n_training_anomalies = math.ceil(xn_train.shape[0] * anomalies_train_size)
    idx = idx[:n_training_anomalies]

    x_anomalous_train = x_anomalous[idx, :]
    y_anomalous_train = y_anomalous[idx]

    x_anomalous_test = np.delete(x_anomalous, idx, axis=0)
    y_anomalous_test = np.delete(y_anomalous, idx)

    x_train = np.vstack([xn_train, x_anomalous_train])
    y_train = np.hstack([yn_train, y_anomalous_train])
    x_test = np.vstack([xn_test, x_anomalous_test])
    y_test = np.hstack([yn_test, y_anomalous_test])
    return x_train, x_test, y_train, y_test


def threshold_score(y_true, score, score_f):
    """Compute the performance of using the i:th score

    The scores are typically computed using an outlier detection algorithm

    Parameters
    ----------
    y_true : array-like
        The true labels

    score : array-like
        The scores

    score_f : callable
        Function for estimating the performance of the i:th scoring

    Returns
    -------
    score : ndarray
        performance for each score as threshold

    See Also
    --------
    wildboar.ensemble.IsolationShapeletForest : an isolation forest for time series

    Examples
    --------
    Setting the offset that maximizes balanced accuracy of a shapelet isolation forest

    >>> from wildboar.ensemble import IsolationShapeletForest
    >>> from wildboar.datasets import load_two_lead_ecg
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> x, y = load_two_lead_ecg()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     x, y, 1, test_size=0.2, anomalies_train_size=0.05
    ... )
    >>> f = IsolationShapeletForest()
    >>> f.fit(x_train)
    >>> scores = f.score_samples(x_train)
    >>> perf = threshold_score(y_train, scores, balanced_accuracy_score)
    >>> f.offset_ = score[np.argmax(perf)]
    """
    ba_score = np.empty(score.shape[0], dtype=float)
    score_copy = np.empty(score.shape[0], dtype=float)
    is_inlier = np.ones(score.shape[0])
    for i in range(score.shape[0]):
        score_copy[:] = score
        is_inlier[:] = 1
        is_inlier[score_copy - score[i] < 0] = -1
        ba_score[i] = score_f(y_true, is_inlier)
    return ba_score
