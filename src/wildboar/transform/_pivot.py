# Authors: Isak Samsten
# License: BSD 3 clause
import numbers

from sklearn.utils.validation import check_scalar

from ..tree._ptree import make_metrics
from ._cpivot import PivotFeatureEngineer
from .base import BaseFeatureEngineerTransform


class PivotTransform(BaseFeatureEngineerTransform):
    """A transform using pivot time series and sampled distance metrics."""

    def __init__(
        self, n_pivots=1, *, metric_factories=None, random_state=None, n_jobs=None
    ):
        """

        Parameters
        ----------

        metric_factories : dict, optional
            The distance metrics. A dictionary where key is:

            - if str, a named distance factory (See ``_DISTANCE_FACTORIES.keys()``)
            - if callable, a function returning a list of ``DistanceMeasure``-objects

            and where value is a dict of parameters to the factory.
        """
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.n_pivots = n_pivots
        self.metric_factories = metric_factories

    def _get_feature_engineer(self):
        if self.metric_factories is None:
            metric_factories = {
                "euclidean": None,
                "dtw": {"min_r": 0, "max_r": 0.25, "n": 20},
            }
        elif isinstance(self.metric_factories, dict):
            metric_factories = self.metric_factories
        else:
            raise TypeError(
                "metric_factories must be dict, got %r" % self.metric_factories
            )

        metrics, _ = make_metrics(metric_factories)
        return PivotFeatureEngineer(
            check_scalar(self.n_pivot, "n_pivots", numbers.Integral, min_val=1),
            metrics,
        )
