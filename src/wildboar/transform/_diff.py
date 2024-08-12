import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator
from ..utils.validation import _check_ts_array
from ._attribute_transform import derivative_transform


class FftTransform(TransformerMixin, BaseEstimator):
    """
    Discrete Fourier Transform.
    """

    def fit(self, X, y=None):
        """
        Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timesteps)
            The samples.
        y : ignore, optional
            Ignored.

        Returns
        -------
        self
            This instance.
        """
        self._validate_data(X, allow_3d=True)
        return self

    def transform(self, X):
        """
        Transform the input.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timesteps)
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples, n_dims, n_timesteps)
            The transformed data.
        """
        X = self._validate_data(X, allow_3d=True, reset=False)
        return np.abs(np.fft.rfft(X))


class DiffTransform(TransformerMixin, BaseEstimator):
    def __init__(self, order=1):
        self.order = order

    def fit(self, X, y=None):
        self._validate_data(X, ensure_min_timesteps=2, allow_3d=True)
        self.order_ = self.order
        return self

    def transform(self, X):
        self._validate_data(X, allow_3d=True, reset=False)
        check_is_fitted(self)
        return np.diff(X, n=self.order_, axis=-1)


class DerivativeTransform(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self._validate_data(X, allow_3d=True)
        return self

    def transform(self, X):
        X = self._validate_data(X, allow_3d=True, reset=False)
        X_t = derivative_transform(_check_ts_array(X))

        if X.ndim == 2:
            return np.squeeze(X_t)
        else:
            return X_t
