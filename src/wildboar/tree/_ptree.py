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
import sys
import warnings

import numpy as np

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
from ..tree.base import BaseTree, TreeClassifierMixin
from ..utils.data import check_dataset

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


def dtw_factory(min_r=0, max_r=0.25, n=10):
    return [_DISTANCE_MEASURE["dtw"](r=r) for r in np.linspace(min_r, max_r, n)]


def ddtw_factory(min_r=0, max_r=0.25, n=10):
    return [_DISTANCE_MEASURE["ddtw"](r=r) for r in np.linspace(min_r, max_r, n)]


def wdtw_factory(min_g=0.05, max_g=0.2, n=10):
    return [_DISTANCE_MEASURE["wdtw"](g=g) for g in np.linspace(min_g, max_g, n)]


_METRIC_FACTORIES = {
    "euclidean": euclidean_factory,
    "dtw": dtw_factory,
    "ddtw": ddtw_factory,
    "wdtw": wdtw_factory,
}


def make_metrics(metric_factories=None):
    if metric_factories is None:
        metric_factories = {
            "euclidean": None,
            "dtw": {"min_r": 0, "max_r": 0.25, "n": 20},
            "ddtw": {"min_r": 0, "max_r": 0.25, "n": 20},
            "wdtw": {"min_g": 0.05, "max_g": 0.0},
        }

    distance_measures = []
    weights = []
    weight = 1.0 / len(metric_factories)
    for metric_factory, metric_factory_params in metric_factories.items():
        if callable(metric_factory):
            metrics = metric_factory(**(metric_factory_params) or {})
        elif metric_factory not in _METRIC_FACTORIES:
            raise ValueError("metric (%r) is not supported" % metric_factory)

        metrics = _METRIC_FACTORIES[metric_factory](**(metric_factory_params or {}))
        for distance_measure in metrics:
            distance_measures.append(distance_measure)
            weights.append(weight / len(metrics))

    return distance_measures, np.array(weights, dtype=np.double)


class ProximityTreeClassifier(TreeClassifierMixin, BaseTree):
    """A classifier that uses a k-branching tree based on pivot-time series.

    Examples
    --------
    >>> from wildboar.datasets import load_dataset
    >>> from wildboar.tree import ProximityTreeClassifier
    >>> x, y = load_dataset("GunPoint")
    >>> f = ProximityTreeClassifier(n_pivot=10, criterion="gini")
    >>> f.fit(x, y)

    References
    ----------
    Lucas, Benjamin, Ahmed Shifaz, Charlotte Pelletier, Lachlan O'Neill, Nayyar Zaidi, \
    Bart Goethals, François Petitjean, and Geoffrey I. Webb. (2019)
        Proximity forest: an effective and scalable distance-based classifier for time
        series. Data Mining and Knowledge Discovery

    """

    def __init__(
        self,
        n_pivot=1,
        *,
        criterion="entropy",
        pivot_sample="label",
        metric_sample="weighted",
        metric_factories=None,
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

        metric_factories : dict, optional
            The distance metrics. A dictionary where key is:

            - if str, a named distance factory (See ``_DISTANCE_FACTORIES.keys()``)
            - if callable, a function returning a list of ``DistanceMeasure``-objects

            and where value is a dict of parameters to the factory.

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

    def _fit(self, x, y, sample_weights, random_state):
        max_depth = (
            sys.getrecursionlimit() if self.max_depth is None else self.max_depth
        )
        if self.min_impurity_decrease < 0.0:
            raise ValueError(
                "min_impurity_decrease must be larger than or equal to 0.0"
            )

        if max_depth <= 0:
            raise ValueError("max_depth must be larger than 0")
        elif max_depth > sys.getrecursionlimit():
            warnings.warn("max_depth exceeds the maximum recursion limit.")

        if self.criterion not in _CLF_CRITERION:
            raise ValueError("criterion (%r) is not supported" % self.criterion)

        if self.pivot_sample not in _PIVOT_SAMPLER:
            raise ValueError("pivot_sample (%r) is not supported" % self.pivot_sample)

        if self.metric_sample not in _DISTANCE_MEASURE_SAMPLER:
            raise ValueError("metric_sample (%r) is not supported" % self.metric_sample)

        distance_measures, weights = make_metrics(
            metric_factories=self.metric_factories
        )
        criterion = _CLF_CRITERION[self.criterion](y, self.n_classes_)
        pivot_sampler = _PIVOT_SAMPLER[self.pivot_sample]()
        distance_measure_sampler = _DISTANCE_MEASURE_SAMPLER[self.metric_sample](
            len(distance_measures), weights=weights
        )
        tree = Tree(distance_measures, self.n_classes_)
        n_features = self.n_pivot

        x = check_dataset(x)
        tree_builder = TreeBuilder(
            x,
            sample_weights,
            pivot_sampler,
            distance_measure_sampler,
            criterion,
            tree,
            random_state,
            max_depth=max_depth,
            n_features=n_features,
        )
        tree_builder.build_tree()
        self.tree_ = tree_builder.tree
