import numpy as np
from sklearn.base import TransformerMixin, _fit_context
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator
from ..utils.validation import _check_ts_array
from ._attribute_transform import (
    backward_derivative_transform,
    central_derivative_transform,
    slope_derivative_transform,
)


class FftTransform(TransformerMixin, BaseEstimator):
    """
    Discrete Fourier Transform.

    Parameters
    ----------
    spectrum : {"amplitude", "phase"}, optional
       The spectrum of FFT transformation.
    """

    _parameter_constraints = {
        "spectrum": [StrOptions({"amplitude", "phase"})],
    }

    def __init__(self, spectrum="amplitude"):
        self.spectrum = spectrum

    @_fit_context(prefer_skip_nested_validation=True)
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
        self.spectrum_ = self.spectrum
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
        check_is_fitted(self)
        X = self._validate_data(X, allow_3d=True, reset=False)

        fft = np.fft.rfft(X)
        if self.spectrum_ == "amplitude":
            return np.abs(fft)
        else:
            return np.angle(fft)


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


_DERIVATIVE_TRANSFORM = {
    "slope": slope_derivative_transform,
    "central": central_derivative_transform,
    "backward": backward_derivative_transform,
}


class DerivativeTransform(TransformerMixin, BaseEstimator):
    """
    Perform derivative transformation on time series data.

    Parameters
    ----------
    method : str, optional
        The method to use for the derivative transformation. Must be one of:
        "slope", "central", or "backward".

        - "backward", computes the derivative at each point using the
          difference between the current and previous elements.
        - "central", computes the derivative at each point using the average of
          the differences between the next and previous elements.
        - "slope", computes a smoothed derivative at each point by averaging
          the difference between the current and previous elements with half
          the difference between the next and previous elements.
    """

    _parameter_constraints = {"method": [StrOptions(_DERIVATIVE_TRANSFORM.keys())]}

    def __init__(self, method="slope"):
        self.method = method

    @_fit_context(prefer_skip_nested_validation=True)
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
        self.method_ = self.method
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
        check_is_fitted(self)
        X = self._validate_data(X, allow_3d=True, reset=False)
        X_t = _DERIVATIVE_TRANSFORM[self.method_](_check_ts_array(X))

        if X.ndim == 2:
            return np.squeeze(X_t)
        else:
            return X_t
