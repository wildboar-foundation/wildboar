import numbers

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils._param_validation import Interval

from ..base import BaseEstimator
from ._distance import pairwise_distance


class KNeighbourClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifier implementing k-nearest neighbours

    Parameters
    ----------
    n_neighbours : int, optional
        The number of neighbours.
    metric : str, optional
        The distance metric.
    metric_params : dict, optional
        Optional parameters to the distance metric.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_metrics>`.

    Attributes
    ----------
    classes_ : ndarray of shapel (n_classes, )
        Known class labels.
    """

    _parameter_constraints: dict = {
        "n_neighbours": Interval(numbers.Integral, 1, None, closed="left"),
        "metric": str,
        "metric_params": [str, None],
    }

    def __init__(self, n_neighbours=5, *, metric="euclidean", metric_params=None):
        self.n_neighbours = n_neighbours
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, x, y):
        """
        Fit the classifier to the training data.

        Parameters
        ----------
        x : univariate time-series or multivaraite time-series
            The input samples.
        y : array-like of shape (n_samples, )
            The input labels.

        Returns
        -------
        KNeighbourClassifier
            This instance.
        """
        self._validate_params()
        x, y = self._validate_data(x, y, allow_3d=True)
        self._fit_X = x.copy()  # Align naming with sklearn
        self._y = y.copy()
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, x):
        """
        Compute probability estimates for the samples in x.

        Parameters
        ----------
        x : univariate time-series or multivariate time-series
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples, len(self.classes_))
            The probability of each class for each sample.
        """
        x = self._validate_data(x, allow_3d=True, reset=False)
        dists = pairwise_distance(
            x,
            self._fit_X,
            dim="mean",
            metric=self.metric,
            metric_params=self.metric_params,
        )
        preds = self._y[
            np.argpartition(dists, self.n_neighbours, axis=1)[:, : self.n_neighbours]
        ]

        probs = np.empty((x.shape[0], len(self.classes_)), dtype=float)
        for i, c in enumerate(self.classes_):
            probs[:, i] = np.sum(preds == c, axis=1) / self.n_neighbours
        return probs

    def predict(self, x):
        """
        Compute the class label for the samples in x.

        Parameters
        ----------
        x : univariate time-series or multivariate time-series
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples, )
            The class label for each sample.
        """
        return np.take(self.classes_, np.argmax(self.predict_proba(x), axis=1))
