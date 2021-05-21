# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten
import numpy as np
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from ._embed_fast import (
    feature_embedding_fit,
    feature_embedding_fit_transform,
    feature_embedding_transform,
)

__all__ = [
    "BaseEmbedding",
]


class BaseEmbedding(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """Base embedding

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
        """Fit the embedding.

        Parameters
        ----------
        x : array-like of shape [n_samples, n_timestep] or [n_samples, n_dimensions, n_timestep] # noqa: E501
            The time series dataset.

        y : None, optional
            For compatibility.

        Returns
        -------
        embedding : self
        """
        x = check_array(x)
        random_state = check_random_state(self.random_state)
        self.n_timestep_ = x.shape[-1]
        self.embedding_ = feature_embedding_fit(
            self._get_feature_engineer(), x, random_state, self.n_jobs
        )
        return self

    def transform(self, x):
        """Transform the dataset.

        Parameters
        ----------
        x : array-like of shape [n_samples, n_timestep] or [n_samples, n_dimensions, n_timestep] # noqa: E501
            The time series dataset.

        Returns
        -------
        x_embedding : ndarray of shape [n_samples, n_outputs]
            The embedding.
        """
        check_is_fitted(self, attributes="embedding_")
        x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions")
        return feature_embedding_transform(self.embedding_, x, self.n_jobs)

    def fit_transform(self, x, y=None):
        """Fit the embedding and return the transform of x.

        Parameters
        ----------
        x : array-like of shape [n_samples, n_timestep] or [n_samples, n_dimensions, n_timestep] # noqa: E501
            The time series dataset.

        y : None, optional
            For compatibility.

        Returns
        -------
        x_embedding : ndarray of shape [n_samples, n_outputs]
            The embedding.
        """
        x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions")

        random_state = check_random_state(self.random_state)
        self.n_timestep_ = x.shape[-1]
        embedding, x_out = feature_embedding_fit_transform(
            self._get_feature_engineer(), x, random_state, self.n_jobs
        )
        self.embedding_ = embedding
        return x_out

    @abstractmethod
    def _get_feature_engineer(self):
        pass
