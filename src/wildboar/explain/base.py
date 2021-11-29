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

# Authors: Isak Samsten

import abc

from sklearn.base import BaseEstimator

__all__ = ["BaseExplanation", "BaseImportance"]


class BaseExplanation(BaseEstimator, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def plot(**kwargs):
        """Plot the explanation

        Returns
        -------
        ax : Axes
            The axes object
        """
        pass


class BaseImportance(BaseExplanation):
    @abc.abstractmethod
    def fit(self, estimator, x, y=None, sample_weight=None):
        """Fit the importance explanation to the estimator

        Parameters
        ----------
        estimator : Estimator
            An estimator
        x : array-like of shape (n_samples, n_timestep) or (n_samples, n_dim, n_timestep) # noqa: E501
            The samples
        y : array-like of shape (n_samples, ), optional
            The labels, by default None
        sample_weight : array-like of shape (n_samples, ), optional
            The sample weights, by default None
        """
        pass
