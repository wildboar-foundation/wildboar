# Authors: Isak Samsten
# License: BSD 3 clause
import numbers

from sklearn.utils._param_validation import Interval, StrOptions

from ._cpivot import PivotFeatureEngineer
from .base import BaseFeatureEngineerTransform


class PivotMixin:

    _parameter_constraints: dict = {
        "n_pivots": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric_factories": [StrOptions({"default"}), dict, list],
    }

    def _get_feature_engineer(self, n_samples):
        from ..tree._ptree import make_metrics

        if (
            isinstance(self.metric_factories, str)
            and self.metric_factories == "default"
        ):
            metric_factories = {
                "euclidean": None,
                "dtw": None,
                "rdtw": {"min_r": 0, "max_r": 0.25, "n": 20},
            }
        elif isinstance(self.metric_factories, list):
            metric_factories = {key: None for key in self.metric_factories}
        else:
            metric_factories = self.metric_factories

        # TODO: weighted sampling?
        metrics, _ = make_metrics(metric_factories)
        return PivotFeatureEngineer(self.n_pivots, metrics)


class PivotTransform(PivotMixin, BaseFeatureEngineerTransform):
    """A transform using pivot time series and sampled distance metrics."""

    _parameter_constraints: dict = {
        **PivotMixin._parameter_constraints,
        **BaseFeatureEngineerTransform._parameter_constraints,
    }

    def __init__(
        self, n_pivots=1, *, metric_factories="default", random_state=None, n_jobs=None
    ):
        """

        Parameters
        ----------

        metric_factories : dict, optional
            The distance metrics. A dictionary where key is:

            - if str, a named distance factory (See ``_DISTANCE_FACTORIES.keys()``)
            - if callable, a function returning a list of ``Metric``-objects

            and where value is a dict of parameters to the factory.
        """
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.n_pivots = n_pivots
        self.metric_factories = metric_factories
