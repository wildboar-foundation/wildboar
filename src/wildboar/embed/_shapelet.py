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

from ._shapelet_fast import RandomShapeletFeatureEngineer
from .base import BaseEmbedding


class RandomShapeletEmbedding(BaseEmbedding):
    """Embed a time series as the distances to a selection of random shapelets.

    Attributes
    ----------
    embedding_ : Embedding
        The underlying embedding object.

    References
    ----------
    Wistuba, Martin, Josif Grabocka, and Lars Schmidt-Thieme.
        Ultra-fast shapelets for time series classification.
        arXiv preprint arXiv:1503.05018 (2015).
    """

    def __init__(
        self,
        n_shapelets=1000,
        *,
        metric="euclidean",
        metric_params=None,
        min_shapelet_size=0,
        max_shapelet_size=1.0,
        n_jobs=None,
        random_state=None
    ):
        """
        Parameters
        ----------
        n_shapelets : int, optional
            The number of shapelets in the resulting embedding

        metric : str, optional
            The distance metric

            - if str use optimized implementations of the named distance measure
            - if callable a function taking two arrays as input

        metric_params: dict, optional
            Parameters to the metric

            - 'euclidean' and 'scaled_euclidean' take no parameters
            - 'dtw' and 'scaled_dtw' take a single paramater 'r'. If 'r' <= 1 it
            is interpreted as a fraction of the time series length. If > 1 it
            is interpreted as an exact time warping window. Use 'r' == 0 for
            a widow size of exactly 1.

        min_shapelet_size : float, optional
            Minimum shapelet size.

        max_shapelet_size : float, optional
            Maximum shapelet size.

        n_jobs : int, optional
            The number of jobs to run in parallel. None means 1 and
            -1 means using all processors.

        random_state : int or RandomState, optional
            The psudo-random number generator.
        """
        super().__init__(n_jobs=n_jobs)
        self.n_shapelets = n_shapelets
        self.metric = metric
        self.metric_params = metric_params
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.random_state = random_state

    def _get_feature_engineer(self):
        if (
            self.min_shapelet_size < 0
            or self.min_shapelet_size > self.max_shapelet_size
        ):
            raise ValueError(
                "`min_shapelet_size` {0} <= 0 or {0} > {1}".format(
                    self.min_shapelet_size, self.max_shapelet_size
                )
            )
        if self.max_shapelet_size > 1:
            raise ValueError(
                "`max_shapelet_size` {0} > 1".format(self.max_shapelet_size)
            )
        max_shapelet_size = int(self.n_timestep_ * self.max_shapelet_size)
        min_shapelet_size = int(self.n_timestep_ * self.min_shapelet_size)
        if min_shapelet_size < 2:
            min_shapelet_size = 2
        return RandomShapeletFeatureEngineer(
            self.n_timestep_,
            self.metric,
            self.metric_params,
            min_shapelet_size,
            max_shapelet_size,
            self.n_shapelets,
        )
