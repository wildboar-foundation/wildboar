# Authors: Isak Samsten
# License: BSD 3 clause

import numbers

import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions

from ..distance import _DISTANCE_MEASURE
from ..tree._cptree import (
    EntropyCriterion,
    GiniCriterion,
    LabelPivotSampler,
    Tree,
    TreeBuilder,
    UniformDistanceMeasureSampler,
    UniformPivotSampler,
    WeightedDistanceMeasureSampler,
)
from ..tree.base import BaseTree, BaseTreeClassifier
from ..utils.validation import check_option

_CLF_CRITERION = {
    "gini": GiniCriterion,
    "entropy": EntropyCriterion,
}
_PIVOT_SAMPLER = {
    "label": LabelPivotSampler,
    "uniform": UniformPivotSampler,
}
_DISTANCE_MEASURE_SAMPLER = {
    "uniform": UniformDistanceMeasureSampler,
    "weighted": WeightedDistanceMeasureSampler,
}


def euclidean_factory():
    return [_DISTANCE_MEASURE["euclidean"]()]


def dtw_factory():
    return [_DISTANCE_MEASURE["dtw"](r=1.0)]


def ddtw_factory():
    return [_DISTANCE_MEASURE["ddtw"](r=1.0)]


def rdtw_factory(min_r=0, max_r=0.25, n=10):
    return [_DISTANCE_MEASURE["dtw"](r=r) for r in np.linspace(min_r, max_r, n)]


def rddtw_factory(min_r=0, max_r=0.25, n=10):
    return [_DISTANCE_MEASURE["ddtw"](r=r) for r in np.linspace(min_r, max_r, n)]


def wdtw_factory(min_g=0.0, max_g=1.0, n=10):
    return [_DISTANCE_MEASURE["wdtw"](g=g) for g in np.linspace(min_g, max_g, n)]


def wddtw_factory(min_g=0.0, max_g=1.0, n=10):
    return [_DISTANCE_MEASURE["wddtw"](g=g) for g in np.linspace(min_g, max_g, n)]


def erp_factory(min_g=0.0, max_g=1.0, n=10):
    return [_DISTANCE_MEASURE["erp"](g=g) for g in np.linspace(min_g, max_g, n)]


def lcss_factory(min_r=0.0, max_r=0.25, min_epsilon=0, max_epsilon=1.0, n=10):
    return [
        _DISTANCE_MEASURE["lcss"](r=r, epsilon=epsilon)
        for epsilon in np.linspace(min_epsilon, max_epsilon, n)
        for r in np.linspace(min_r, max_r, n)
    ]


def msm_factory(min_c=0.01, max_c=100, n=10):
    return [_DISTANCE_MEASURE["msm"](c=c) for c in np.linspace(min_c, max_c, n)]


def twe_factory(
    min_penalty=0.00001, max_penalty=1.0, min_stiffness=10e-5, max_stiffness=0.1, n=10
):
    return [
        _DISTANCE_MEASURE["twe"](penalty=penalty, stiffness=stiffness)
        for penalty in np.linspace(min_penalty, max_penalty, n)
        for stiffness in np.linspace(min_stiffness, max_stiffness, n)
    ]


_METRIC_FACTORIES = {
    "euclidean": euclidean_factory,
    "dtw": dtw_factory,
    "ddtw": ddtw_factory,
    "rdtw": rdtw_factory,
    "rddtw": rddtw_factory,
    "wdtw": wdtw_factory,
    "wddtw": wddtw_factory,
    "erp": erp_factory,
    "lcss": lcss_factory,
    "msm": msm_factory,
    "twe": twe_factory,
}


def make_metrics(metric_factories=None):
    distance_measures = []
    weights = []
    weight = 1.0 / len(metric_factories)
    for metric_factory, metric_factory_params in metric_factories.items():
        if callable(metric_factory):
            metrics = metric_factory(**(metric_factory_params) or {})
        else:
            metrics = check_option(
                _METRIC_FACTORIES,
                metric_factory,
                "metric_factories key",
            )(**(metric_factory_params or {}))

        for distance_measure in metrics:
            distance_measures.append(distance_measure)
            weights.append(weight / len(metrics))

    return distance_measures, np.array(weights, dtype=np.double)


class ProximityTreeClassifier(BaseTreeClassifier):
    """A classifier that uses a k-branching tree based on pivot-time series.

    Examples
    --------
    >>> from wildboar.datasets import load_dataset
    >>> from wildboar.tree import ProximityTreeClassifier
    >>> x, y = load_dataset("GunPoint")
    >>> f = ProximityTreeClassifier(
    ...     n_pivot=10,
    ...     metric_factories={
    ...         "rdtw": {"min_r": 0.1, "max_r": 0.25},
    ...         "msm": {"min_c": 0.1, "max_c": 100, "n": 20}
    ...     },
    ...     criterion="gini"
    ... )
    >>> f.fit(x, y)

    References
    ----------
    Lucas, Benjamin, Ahmed Shifaz, Charlotte Pelletier, Lachlan O'Neill, Nayyar Zaidi, \
    Bart Goethals, François Petitjean, and Geoffrey I. Webb. (2019)
        Proximity forest: an effective and scalable distance-based classifier for time
        series. Data Mining and Knowledge Discovery

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
            StrOptions(_DISTANCE_MEASURE_SAMPLER.keys()),
        ],
        "metric_factories": [
            StrOptions({"default"}),
            list,
            dict,
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
        metric_factories="default",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        class_weight=None,
        random_state=None,
    ):
        """
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

        metric_factories : "default", list or dict, optional
            The distance metrics.

            If dict, a dictionary where key is:

            - if str, a named distance factory (See ``_DISTANCE_FACTORIES.keys()``)
            - if callable, a function returning a list of ``DistanceMeasure``-objects

            and where value is a dict of parameters to the factory.

            If list, a list of named factories or callables.

            If "default", use the parameterization of (Lucas et.al, 2019)

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
        """
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
        self.metric_factories = metric_factories
        self.class_weight = class_weight
        self.random_state = random_state

    def _fit(self, x, y, sample_weights, max_depth, random_state):
        if self.metric_factories == "default":
            std_x = np.std(x)
            metric_factories = {
                "euclidean": None,
                "dtw": None,
                "ddtw": None,
                "rdtw": {"min_r": 0, "max_r": 0.25, "n": 50},
                "rddtw": {"min_r": 0, "max_r": 0.25, "n": 50},
                "wdtw": {"min_g": std_x * 0.2, "max_g": std_x, "n": 50},
                "wddtw": {"min_g": std_x * 0.2, "max_g": std_x, "n": 50},
                "lcss": {
                    "min_r": 0,
                    "max_r": 0.25,
                    "min_epsilon": std_x * 0.2,
                    "max_epsilon": std_x,
                    "n": 20,
                },
                "erp": {"min_g": 0, "max_g": 1.0, "n": 50},
                "msm": {"min_c": 0.01, "max_c": 100, "n": 50},
                "twe": {
                    "min_penalty": 0.00001,
                    "max_penalty": 1.0,
                    "min_stiffness": 0.000001,
                    "max_stiffness": 0.1,
                    "n": 20,
                },
            }
        elif isinstance(self.metric_factories, list):
            metric_factories = {key: None for key in self.metric_factories}
        else:
            metric_factories = self.metric_factories

        distance_measures, weights = make_metrics(metric_factories=metric_factories)
        Criterion = _CLF_CRITERION[self.criterion]
        PivotSampler = _PIVOT_SAMPLER[self.pivot_sample]
        DistanceMeasureSampler = _DISTANCE_MEASURE_SAMPLER[self.metric_sample]
        tree_builder = TreeBuilder(
            x,
            sample_weights,
            PivotSampler(),
            DistanceMeasureSampler(len(distance_measures), weights=weights),
            Criterion(y, self.n_classes_),
            Tree(distance_measures, self.n_classes_),
            random_state,
            n_features=self.n_pivot,
            max_depth=max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        tree_builder.build_tree()
        self.tree_ = tree_builder.tree
