# Authors: Isak Samsten
# License: BSD 3 clause

import math

import numpy as np
from sklearn import model_selection
from sklearn.utils import check_random_state
from sklearn.utils._random import sample_without_replacement


def outlier_train_test_split(
    x, y, normal_class, test_size=0.2, anomalies_train_size=0.05, random_state=None
):
    """
    Outlier training and testing split from classification dataset.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_timestep) or (n_samples, n_dim, n_timestep)
        Input data samples.
    y : array-like of shape (n_samples,)
        Input class label.
    normal_class : int
        Class label that should be considered as the normal class.
    test_size : float, optional
        Size of the test set.
    anomalies_train_size : float, optional
        Contamination of anomalies in the training dataset.
    random_state : int or RandomState, optional
        Psudo random state used for stable results.

    Returns
    -------
    x_train : array-like
        Training samples.
    x_test : array-like
        Test samples.
    y_train : array-like
        Training labels (either 1 or -1, where 1 denotes normal and -1 anomalous).
    y_test : array-like
        Test labels (either 1 or -1, where 1 denotes normal and -1 anomalous).

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
