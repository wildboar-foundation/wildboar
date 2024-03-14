import numpy as np
from scipy.sparse import csr_array
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator
from ..distance import argmin_distance


class SegmenterMixin:
    _estimator_type = "segmenter"

    def fit_predict(self, X, y=None):
        if y is None:
            return self.fit(X).predict(X)
        else:
            return self.fit(X, y).predict(X)


class BaseSegmenter(TransformerMixin, BaseEstimator):
    """
    Base class for segmenters.

    Inheriting classes must set the ``labels_`` attribute to a list of lists
    with the index position of the segments.

    Parameters
    ----------
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.

    Attributes
    ----------
    labels_ : list of shape (n_samples, )
        A list of n_samples lists with the start index of the segment.
    """

    def __init__(self, metric="euclidean", metric_params=None):
        self.metric = metric
        self.metric_params = metric_params

    def predict(self, X):
        """
        Predict the position with the change point.

        The predicted segmentation is based on the  closest sample from the
        training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            The input data.

        Returns
        -------
        csr_array of shape (n_samples, n_timesteps)
            A boolean array with the start of the change point set to True.
        """
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        rowind = []
        colind = []
        data = []

        for i in range(X.shape[0]):
            closest = argmin_distance(
                X[i], self._fit_X, metric=self.metric, metric_params=self.metric_params
            )
            labels = self.labels_[closest[0, 0]]

            for start in labels:
                rowind.append(i)
                colind.append(start)
                data.append(1)

        return csr_array(
            (data, (rowind, colind)),
            shape=(X.shape[0], self.n_timesteps_in_),
            dtype=bool,
        )

    def transform(self, X):
        """
        Transform X such that each segment is labeled with a unique label.

        The predicted segmentation is based on the  closest sample from the
        training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            The input data.

        Returns
        -------
        ndarray of shape (n_samples, n_timesteps)
            An array with the segments annotated with a label.
        """
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        X_out = np.zeros((X.shape[0], self.n_timesteps_in_), dtype=float)

        for i in range(X.shape[0]):
            current_label = 1
            closest = argmin_distance(
                X[i], self._fit_X, metric=self.metric, metric_params=self.metric_params
            )
            labels = self.labels_[closest[0, 0]]

            for start in labels:
                X_out[i, start:] = current_label
                current_label += 1

        return X_out
