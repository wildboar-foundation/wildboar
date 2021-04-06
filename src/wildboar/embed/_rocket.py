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

from sklearn.utils.validation import check_random_state
from .base import BaseEmbedding
from ._rocket_fast import RocketFeatureEngineer


class RocketEmbedding(BaseEmbedding):
    """Embedd a time series using random convolution features

    Attributes
    ----------
    embedding_ : Embedding
        The underlying embedding

    References
    ----------
    Dempster, Angus, François Petitjean, and Geoffrey I. Webb.
        ROCKET: exceptionally fast and accurate time series classification using
        random convolutional kernels.
        Data Mining and Knowledge Discovery 34.5 (2020): 1454-1495.
    """

    def __init__(self, n_kernels=1000, *, n_jobs=None, random_state=None):
        """
        Parameters
        ----------
        n_kernels : int, optional
            The number of kernels.

        n_jobs : int, optional
            The number of jobs to run in parallel. None means 1 and
            -1 means using all processors.

        random_state : int or RandomState, optional
            The psuodo-random number generator.
        """
        super().__init__(n_jobs=n_jobs)
        self.n_kernels = n_kernels
        self.random_state = random_state

    def _get_feature_engineer(self):
        return RocketFeatureEngineer(self.n_kernels, [7, 9, 11])
