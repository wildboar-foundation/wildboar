# Authors: Isak Samsten
# License: BSD 3 clause

from ..distance import _SUBSEQUENCE_DISTANCE_MEASURE
from ._shapelet_fast import RandomShapeletFeatureEngineer
from .base import BaseFeatureEngineerTransform


class RandomShapeletTransform(BaseFeatureEngineerTransform):
    """Transform a time series to the distances to a selection of random shapelets.

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
            The number of shapelets in the resulting transform

        metric : str, optional
            Distance metric used to identify the best shapelet.

            See ``distance._SUBSEQUENCE_DISTANCE_MEASURE.keys()`` for a list of
            supported metrics.

        metric_params : dict, optional
            Parameters for the distance measure.

            Read more about the parameters in the
            :ref:`User guide <list_of_subsequence_metrics>`.

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
        max_shapelet_size = int(self.n_timesteps_in_ * self.max_shapelet_size)
        min_shapelet_size = int(self.n_timesteps_in_ * self.min_shapelet_size)
        if min_shapelet_size < 2:
            # NOTE: To ensure that the same random_seed generates the same shapelets
            # in future versions we keep the limit of 2 timesteps for a shapelet as long
            # as the time series is at least 2 timesteps. Otherwise we fall back to 1
            # timestep.
            #
            # TODO(1.2): consider breaking backwards compatibility and always limit to
            #            1 timestep.
            if self.n_timesteps_in_ < 2:
                min_shapelet_size = 1
            else:
                min_shapelet_size = 2

        distance_measure = _SUBSEQUENCE_DISTANCE_MEASURE.get(self.metric, None)
        if distance_measure is None:
            raise ValueError("invalid distance measure (%r)" % self.metric)
        metric_params = self.metric_params or {}
        return RandomShapeletFeatureEngineer(
            distance_measure(**metric_params),
            min_shapelet_size,
            max_shapelet_size,
            self.n_shapelets,
        )