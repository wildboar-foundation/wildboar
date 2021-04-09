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

import abc

from sklearn.base import BaseEstimator

__all__ = ["BaseCounterfactual"]


class BaseCounterfactual(BaseEstimator):
    """Base estimator for counterfactual explanations"""

    @abc.abstractmethod
    def fit(self, estimator):
        """Fit the counterfactual to a given estimator

        Parameters
        ----------
        estimator : object
            An estimator for which counterfactual explanations are produced

        Returns
        -------
        self
        """
        pass

    @abc.abstractmethod
    def transform(self, x, y):
        """Transform the i:th sample in x to a sample that would be labeled as the i:th label in y

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep) or (n_samples, n_dimension, n_timestep)
            The samples to generate counterfactual explanations for

        y : array-like of shape (n_samples,)
            The desired label of the counterfactual sample

        Returns
        -------

        counterfactuals : ndarray of same shape as x
            The counterfactual for each sample. If success[i] == False, then
            the value of counterfactuals[i] is undefined.

        success : ndarray of shape (n_samples,)
             Boolean vector indicating successful transformations.
        """
        pass
