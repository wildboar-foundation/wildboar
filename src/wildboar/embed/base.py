# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ._embedding import FeatureEngineerEmbedding

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

    def fit(self, x, y=None):
        """Fit the embedding.

        Parameters
        ----------
        x : array-like of shape [n_samples, n_timestep] or [n_samples, n_dimensions, n_timestep]
            The time series dataset.

        y : None, optional
            For compatibility.

        Returns
        -------
        embedding : self
        """
        x = check_array(x)
        self.n_timestep_ = x.shape[-1]
        fee = FeatureEngineerEmbedding(self._get_feature_engineer())
        fee.fit_embedding(x)
        self.embedding_ = fee.embedding_
        return self

    def transform(self, x):
        """Transform the dataset.

        Parameters
        ----------
        x : array-like of shape [n_samples, n_timestep] or [n_samples, n_dimensions, n_timestep]
            The time series dataset.

        Returns
        -------
        x_embedding : ndarray of shape [n_samples, n_outputs]
            The embedding.
        """
        check_is_fitted(self, attributes="embedding_")
        x = check_array(x)
        return self.embedding_.apply(x)

    def fit_transform(self, x, y=None):
        """Fit the embedding and return the transform of x.

        Parameters
        ----------
        x : array-like of shape [n_samples, n_timestep] or [n_samples, n_dimensions, n_timestep]
            The time series dataset.

        y : None, optional
            For compatibility.

        Returns
        -------
        x_embedding : ndarray of shape [n_samples, n_outputs]
            The embedding.
        """
        x = check_array(x)
        self.n_timestep_ = x.shape[-1]
        fee = FeatureEngineerEmbedding(self._get_feature_engineer())
        x_out = fee.fit_embedding_transform(x)
        self.embedding_ = fee.embedding_
        return x_out

    @abstractmethod
    def _get_feature_engineer(self):
        pass