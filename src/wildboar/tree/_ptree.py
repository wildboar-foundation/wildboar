# Authors: Isak Samsten
# License: BSD 3 clause

import numbers
import warnings

import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions

from wildboar.utils.validation import check_option

from ..distance._distance import _METRICS
from ..distance._multi_metric import make_metrics
from ..tree._cptree import (
    EntropyCriterion,
    GiniCriterion,
    LabelPivotSampler,
    Tree,
    TreeBuilder,
    UniformMetricSampler,
    UniformPivotSampler,
    WeightedMetricSampler,
)
from ._base import BaseTree, BaseTreeClassifier

_CLF_CRITERION = {
    "gini": GiniCriterion,
    "entropy": EntropyCriterion,
}

_PIVOT_SAMPLER = {
    "label": LabelPivotSampler,
    "uniform": UniformPivotSampler,
}

_METRICS_SAMPLER = {
    "uniform": UniformMetricSampler,
    "weighted": WeightedMetricSampler,
}

_METRIC_NAMES = set(_METRICS.keys())
_METRIC_NAMES.add("auto")
_METRIC_NAMES.add("default")  # TODO(1.3)
_METRIC_NAMES = frozenset(_METRIC_NAMES)


# noqa: D409
class ProximityTreeClassifier(BaseTreeClassifier):
    """
    A classifier that uses a k-branching tree based on pivot-time series.

    Parameters
    ----------
    n_pivot : int, optional
        The number of pivots to sample at each node.
    criterion : {"entropy", "gini"}, optional
        The impurity criterion.
    pivot_sample : {"label", "uniform"}, optional
        The pivot sampling method.
    metric_sample : {"uniform", "weighted"}, optional
        The metric sampling method.
    metric : {"auto", "default"}, str or list, optional
        The distance metrics. By default, we use the parameterization suggested by
        Lucas et.al (2019).

        - If "auto", use the default metric specification, suggested by
          (Lucas et. al, 2020).
        - If str, use a single metric or default metric specification.
        - If list, custom metric specification can be given as a list of
          tuples, where the first element of the tuple is a metric name and the
          second element a dictionary with a parameter grid specification. A
          parameter grid specification is a `dict` with two mandatory and one
          optional key-value pairs defining the lower and upper bound on the
          values as well as the number of values in the grid. For example, to
          specifiy a grid over the argument 'r' with 10 values in the range 0
          to 1, we would give the following specification: 
          `dict(min_r=0, max_r=1, num_r=10)`.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_metrics>`.
    metric_params : dict, optional
        Parameters for the distance measure. Ignored unless metric is a string.

        Read more about the parameters in the :ref:`User guide
        <list_of_metrics>`.
    metric_factories : dict, optional
        A metric specification.

        .. deprecated:: 1.2
            Use the combination of metric and metric params.
    max_depth : int, optional
        The maximum tree depth.
    min_samples_split : int, optional
        The minimum number of samples to consider a split.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf.
    min_impurity_decrease : float, optional
        The minimum impurity decrease to build a sub-tree.
    class_weight : dict or "balanced", optional
        Weights associated with the labels.

        - if dict, weights on the form {label: weight}.
        - if "balanced" each class weight inversely proportional to the class
            frequency.
        - if None, each class has equal weight.
    random_state : int or RandomState
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

    References
    ----------
    Lucas, Benjamin, Ahmed Shifaz, Charlotte Pelletier, Lachlan O'Neill, Nayyar Zaidi, \
    Bart Goethals, François Petitjean, and Geoffrey I. Webb. (2019)
        Proximity forest: an effective and scalable distance-based classifier for time
        series. Data Mining and Knowledge Discovery

    Examples
    --------
    Fit a single proximity tree, with dynamic time warping and move-split-merge metrics.

    >>> from wildboar.datasets import load_dataset
    >>> from wildboar.tree import ProximityTreeClassifier
    >>> x, y = load_dataset("GunPoint")
    >>> f = ProximityTreeClassifier(
    ...     n_pivot=10,
    ...     metrics=[
    ...         ("dtw", {"min_r": 0.1, "max_r": 0.25}),
    ...         ("msm", {"min_c": 0.1, "max_c": 100, "num_c": 20})
    ...     ],
    ...     criterion="gini"
    ... )
    >>> f.fit(x, y)

    """

    _parameter_constraints: dict = {
        **BaseTree._parameter_constraints,
        "n_pivot": [
            Interval(numbers.Integral, 1, None, closed="left"),
        ],
        "criterion": [
            StrOptions(_CLF_CRITERION.keys()),
        ],
        "pivot_sample": [
            StrOptions(_PIVOT_SAMPLER.keys()),
        ],
        "metric_sample": [
            StrOptions(_METRICS_SAMPLER.keys()),
        ],
        "metric": [
            StrOptions(_METRIC_NAMES, deprecated={"default"}),
            list,
            dict,
        ],
        "metric_params": [None, dict],
        "metric_factories": [  # TODO(1.4)
            StrOptions({"default", "auto"}, deprecated={"default"}),
            list,
            None,
        ],
        "random_state": ["random_state"],
        "class_weight": [
            StrOptions({"balanced"}),
            dict,
            None,
        ],
    }

    def __init__(
        self,
        n_pivot=1,
        *,
        criterion="entropy",
        pivot_sample="label",
        metric_sample="weighted",
        metric="auto",
        metric_params=None,
        metric_factories=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.n_pivot = n_pivot
        self.criterion = criterion
        self.pivot_sample = pivot_sample
        self.metric_sample = metric_sample
        self.metric = metric
        self.metric_params = metric_params
        self.metric_factories = metric_factories
        self.class_weight = class_weight
        self.random_state = random_state

    def _fit(self, x, y, sample_weights, max_depth, random_state):
        if self.metric_factories is not None:
            # TODO(1.4)
            warnings.warn(
                "The parameter metric_factories has been renamed to metric "
                "in 1.2 and will be removed in 1.4",
                DeprecationWarning,
            )
            if isinstance(self.metric_factories, dict):
                metric = list(self.metric_factories.items())

            metric = self.metric_factories

        metric = self.metric
        if isinstance(metric, str) and metric in ["default", "auto"]:
            if metric == "default":
                # TODO(1.4)
                warnings.warn(
                    "The default value for 'metric' has changed to 'auto' in 1.2 and "
                    "'default' will be removed in 1.4.",
                    DeprecationWarning,
                )

            std_x = np.std(x)
            metric_spec = [
                ("euclidean", None),
                ("dtw", None),
                ("ddtw", None),
                ("dtw", {"min_r": 0, "max_r": 0.25, "default_n": 50}),
                ("ddtw", {"min_r": 0, "max_r": 0.25, "default_n": 50}),
                ("wdtw", {"min_g": std_x * 0.2, "max_g": std_x, "default_n": 50}),
                ("wddtw", {"min_g": std_x * 0.2, "max_g": std_x, "default_n": 50}),
                (
                    "lcss",
                    {
                        "min_epsilon": std_x * 0.2,
                        "max_epsilon": std_x,
                        "min_r": 0,
                        "max_r": 0.25,
                        "default_n": 20,
                    },
                ),
                ("erp", {"min_g": 0, "max_g": 1.0, "default_n": 50}),
                ("msm", {"min_c": 0.01, "max_c": 100, "default_n": 50}),
                (
                    "twe",
                    {
                        "min_penalty": 0.00001,
                        "max_penalty": 1.0,
                        "min_stiffness": 0.000001,
                        "max_stiffness": 0.1,
                        "default_n": 20,
                    },
                ),
            ]
            metrics, weights = make_metrics(metric_spec)
        elif isinstance(metric, str) and metric in _METRIC_NAMES:
            Metric = check_option(_METRICS, metric, "metric")
            metrics = [Metric(**(self.metric_params or {}))]
            weights = np.ones(1)
        else:
            metrics, weights = make_metrics(metric)

        Criterion = _CLF_CRITERION[self.criterion]
        PivotSampler = _PIVOT_SAMPLER[self.pivot_sample]
        MetricSampler = _METRICS_SAMPLER[self.metric_sample]
        tree_builder = TreeBuilder(
            x,
            sample_weights,
            PivotSampler(),
            MetricSampler(len(metrics), weights=weights),
            Criterion(y, self.n_classes_),
            Tree(metrics, self.n_classes_),
            random_state,
            n_attributes=self.n_pivot,
            max_depth=max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        tree_builder.build_tree()
        self.tree_ = tree_builder.tree
