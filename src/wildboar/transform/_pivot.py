# Authors: Isak Samsten
# License: BSD 3 clause
import numbers

from sklearn.utils._param_validation import Interval, StrOptions

from ..distance._multi_metric import make_metrics
from ._cpivot import PivotFeatureEngineer
from .base import BaseFeatureEngineerTransform


class PivotMixin:
    _parameter_constraints: dict = {
        "n_pivots": [Interval(numbers.Integral, 1, None, closed="left")],
        "metrics": [StrOptions({"auto"}), list],
    }

    def _get_feature_engineer(self, n_samples):
        if isinstance(self.metrics, str) and self.metrics == "auto":
            metric_specs = [
                ("euclidean", None),
                ("dtw", None),
                ("ddtw", None),
                ("dtw", dict(min_r=0.0, max_r=0.25, num_r=10)),
                ("ddtw", dict(min_r=0.0, max_r=0.25, num_r=10)),
                ("wdtw", dict(min_g=0.2, max_g=1.0)),
                ("wddtw", dict(min_g=0.2, max_g=1.0)),
                (
                    "lcss",
                    dict(
                        min_r=0.0,
                        max_r=0.25,
                        min_epsilon=0.2,
                        max_epsilon=1.0,
                    ),
                ),
                ("erp", dict(min_g=0, max_g=1.0)),
                ("msm", dict(min_c=0.01, max_c=100, num_c=50)),
                (
                    "twe",
                    dict(
                        min_penalty=0.00001,
                        max_penalty=1.0,
                        min_stiffness=0.000001,
                        max_stiffness=0.1,
                    ),
                ),
            ]
        else:
            metric_specs = self.metrics

        # TODO: weighted sampling?
        metrics, _ = make_metrics(metric_specs)
        return PivotFeatureEngineer(self.n_pivots, metrics)


class PivotTransform(PivotMixin, BaseFeatureEngineerTransform):
    """A transform using pivot time series and sampled distance metrics."""

    _parameter_constraints: dict = {
        **PivotMixin._parameter_constraints,
        **BaseFeatureEngineerTransform._parameter_constraints,
    }

    def __init__(
        self,
        n_pivots=100,
        *,
        metrics="auto",
        random_state=None,
        n_jobs=None,
    ):
        """

        Parameters
        ----------

        n_pivot : int, optional
            The number of pivot time series.

        metrics : {'auto'} or list, optional
            - If str, the distance metric used to identify the best shapelet.

            - If list, multiple metrics specified as a list of tuples, where the first
              element of the tuple is a metric name and the second element a dictionary
              with a parameter grid specification. A parameter grid specification is a
              dict with two mandatory and one optional key-value pairs defining the
              lower and upper bound on the values and number of values in the grid. For
              example, to specifiy a grid over the argument 'r' with 10 values in the
              range 0 to 1, we would give the following specification: ``dict(min_r=0,
              max_r=1, num_r=10)``.

            Read more about the metrics and their parameters in the
            :ref:`User guide <list_of_subsequence_metrics>`.

        random_state : int or np.RandomState, optional
            The random state

        n_jobs : int, optional
            The number of cores to use.
        """
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.n_pivots = n_pivots
        self.metrics = metrics
