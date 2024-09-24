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
        ndarray of shape (n_samples, n_dims, m_timesteps)
            The transformed data. If n_timesteps is even m_timesteps is
            (n_timesteps/2) + 1; otherwise (n_timesteps + 1) / 2.
        """
        X = self._validate_data(X, allow_3d=True, reset=False)
        return np.abs(np.fft.rfft(X))


class DiffTransform(TransformerMixin, BaseEstimator):
    """
    A transformer that applies a difference transformation to time series data.

    Parameters
    ----------
    order : int, optional
        The order of the difference operation. Default is 1.
    """

    def __init__(self, order=1):
        self.order = order

    def fit(self, X, y=None):
        """
        Fit the model to the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timesteps)
            The input data to fit the model. Must have at least two timesteps.
        y : array-like, optional
            Not used.

        Returns
        -------
        object
            Returns the instance of the fitted model.
        """
        self._validate_data(X, ensure_min_timesteps=2, allow_3d=True)
        self.order_ = self.order
        return self

    def transform(self, X):
        """
        Transform the input data by computing the discrete differences.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timesteps)
            The input data to be transformed.

        Returns
        -------
        array_like
            An array containing the discrete difference of the input data along
            the last axis.
        """
        self._validate_data(X, allow_3d=True, reset=False)
        check_is_fitted(self)
        return np.diff(X, n=self.order_, axis=-1)


class DerivativeTransform(TransformerMixin, BaseEstimator):
    """
    Perform derivative transformation on time series data.
    """

    def fit(self, X, y=None):
        """
        Fit the model to the provided data.

        Only performs input validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timesteps)
            The input data to fit the model.
        y : array-like, optional
            Not used.

        Returns
        -------
        object
            Returns the instance itself.
        """
        self._validate_data(X, allow_3d=True)
        return self

    def transform(self, X):
        """
        Transform the input data using a derivative transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timesteps)
            The input data to be transformed.

        Returns
        -------
        array
            The transformed data.
        """
        X = self._validate_data(X, allow_3d=True, reset=False)
        X_t = derivative_transform(_check_ts_array(X))

        if X.ndim == 3:
            return np.squeeze(X_t)
        else:
            return X_t
