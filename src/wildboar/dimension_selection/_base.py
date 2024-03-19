import abc
import numbers

import numpy as np
from sklearn.base import TransformerMixin, _fit_context, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval

from ..base import BaseEstimator
from ..distance import pairwise_distance
from ..utils.validation import MetricOptions, check_array


class DimensionSelectorMixin(TransformerMixin, metaclass=abc.ABCMeta):
    """
    Mixin for dimension selector.
    """

    def get_dimensions(self, indices=False):
        """
        Get a boolean mask with the selected dimensions.

        Parameters
        ----------
        indices : bool, optional
            If True, return the indices instead of a boolean mask.

        Returns
        -------
        ndarray of shape (n_selected_dims, )
            An index that selects the retained dimensions.
        """
        check_is_fitted(self)
        mask = self._get_dimensions()
        return mask if not indices else np.flatnonzero(mask)

    @abc.abstractmethod
    def _get_dimensions(self):
        pass

    def transform(self, X):
        """
        Reduce X to the selected dimensions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep)
            The samples.

        Returns
        -------
        ndarray of shape (n_samples, n_selected_dims, n_timestep)
            The samples with only the selected dimensions.
        """
        X = self._validate_data(X, reset=False, allow_3d=True)
        return self._transform(X)

    def _transform(self, X):
        mask = self.get_dimensions()
        X_new = X[:, mask, :]
        if X_new.shape[1] == 1:
            X_new = np.squeeze(X_new, axis=1)
        return X_new

    def inverse_transform(self, X):
        """
        Reverse the transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_selected_dims, n_timestep)
            The samples.

        Returns
        -------
        ndarray of shape (n_samples, n_dims, n_timestep)
            The samples with zeros inserted where dimensions
            would have been removed by :meth:`transform`.
        """
        X = check_array(X, allow_3d=True)

        if X.shape[-1] != self.n_timesteps_in_:
            raise ValueError("incorrect number of timesteps")

        dims = self.get_dimensions()
        if dims.sum() != X.shape[1]:
            raise ValueError("Not the same number of dimensions as when fit")

        X_inv = np.zeros((X.shape[0], dims.shape[0], X.shape[-1]), dtype=X.dtype)
        X_inv[:, dims, :] = X
        return X_inv


class BaseDistanceSelector(
    DimensionSelectorMixin, BaseEstimator, metaclass=abc.ABCMeta
):
    _parameter_constraints = {
        "n_jobs": [None, numbers.Integral],
        "metric": [MetricOptions()],
        "metric_params": [None, dict],
        "sample": [
            None,
            Interval(numbers.Real, 0.0, 1.0, closed="right"),
            Interval(numbers.Integral, 0, None, closed="neither"),
        ],
        "random-state": ["random-state"],
    }

    def __init__(
        self,
        *,
        sample=None,
        metric="euclidean",
        metric_params=None,
        n_jobs=None,
        random_state=None,
    ):
        self.n_jobs = n_jobs
        self.metric = metric
        self.metric_params = metric_params
        self.sample = sample
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Learn the dimensions to select.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims, n_timestep)
            The training samples.
        y : array-like of shape (n_samples, ), optional
            Ignored.

        Returns
        -------
        object
            The instance itself.
        """

        if y is None:
            X = self._validate_data(X, allow_3d=True, ensure_min_dims=2)
        else:
            X, y = self._validate_data(X, y, allow_3d=True, ensure_min_dims=2)

        random_state = check_random_state(self.random_state)
        Y = X
        if self.sample is not None:
            idx = np.arange(X.shape[0])

            random_state.shuffle(idx)
            if isinstance(self.sample, numbers.Real):
                idx = idx[: int(idx.size * self.sample)]
            elif isinstance(self.sample, numbers.Integral):
                if self.sample > X.shape[0]:
                    raise ValueError(
                        "sample cannot be larger than the number of samples"
                    )
                idx = idx[: self.sample]
            else:
                raise ValueError("sample must be int or float")
            Y = X[idx, :, :]

        distance = pairwise_distance(
            X,
            Y,
            dim="full",
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        self._fit(distance, y)
        return self

    @abc.abstractmethod
    def _fit(self, distance, y=None):
        pass
