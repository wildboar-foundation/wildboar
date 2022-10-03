# Authors: Isak Samsten
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from ..base import BaseEstimator
from ._cfeature_transform import (
    feature_transform_fit,
    feature_transform_fit_transform,
    feature_transform_transform,
)

__all__ = [
    "BaseFeatureEngineerTransform",
]


class BaseFeatureEngineerTransform(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """Base feature engineer transform

    Attributes
    ----------

    embedding_ : Embedding
        The underlying embedding.

    """

    def __init__(self, *, random_state=None, n_jobs=None):
        """
        Parameters
        ----------
        n_jobs : int, optional
            The number of jobs to run in parallel. None means 1 and
            -1 means using all processors.
        """
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, x, y=None):
        """Fit the transform.

        Parameters
        ----------
        x : array-like of shape [n_samples, n_timestep] or
        [n_samples, n_dimensions, n_timestep]
            The time series dataset.

        y : None, optional
            For compatibility.

        Returns
        -------
        self : self
        """
        x = self._validate_data(x, allow_3d=True, dtype=np.double)
        self.embedding_ = feature_transform_fit(
            self._get_feature_engineer(), x, check_random_state(self.random_state)
        )
        return self

    def transform(self, x):
        """Transform the dataset.

        Parameters
        ----------
        x : array-like of shape [n_samples, n_timestep] or
        [n_samples, n_dimensions, n_timestep]
            The time series dataset.

        Returns
        -------
        x_transform : ndarray of shape [n_samples, n_outputs]
            The transformation.
        """
        check_is_fitted(self, attributes="embedding_")
        x = self._validate_data(x, reset=False, allow_3d=True, dtype=np.double)
        return feature_transform_transform(self.embedding_, x, self.n_jobs)

    def fit_transform(self, x, y=None):
        """Fit the embedding and return the transform of x.

        Parameters
        ----------
        x : array-like of shape [n_samples, n_timestep] or
        [n_samples, n_dimensions, n_timestep]
            The time series dataset.

        y : None, optional
            For compatibility.

        Returns
        -------
        x_embedding : ndarray of shape [n_samples, n_outputs]
            The embedding.
        """
        x = self._validate_data(x, allow_3d=True, dtype=np.double)
        embedding, x_out = feature_transform_fit_transform(
            self._get_feature_engineer(),
            x,
            check_random_state(self.random_state),
            self.n_jobs,
        )
        self.embedding_ = embedding
        return x_out

    @abstractmethod
    def _get_feature_engineer(self):
        pass
