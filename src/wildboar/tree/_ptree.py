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

from wildboar.distance import _DISTANCE_MEASURE
from wildboar.tree._cptree import (
    EntropyCriterion,
    GiniCriterion,
    LabelPivotSampler,
    Tree,
    TreeBuilder,
    UniformDistanceMeasureSampler,
    UniformPivotSampler,
    WeightedDistanceMeasureSampler,
)
from wildboar.tree.base import BaseTree, TreeClassifierMixin
from wildboar.utils.data import check_dataset
from wildboar.utils.decorators import unstable

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


def make_euclidean():
    return [_DISTANCE_MEASURE["euclidean"]()]


def make_dtw(min_r=0, max_r=0.25, n=10):
    return [_DISTANCE_MEASURE["dtw"](r=r) for r in np.linspace(min_r, max_r, n)]


_METRICS = {"euclidean": make_euclidean, "dtw": make_dtw}


def make_metrics(metrics=None, metrics_params=None):
    if metrics is None:
        metrics = ["euclidean", "dtw"]
    if metrics_params is None:
        metrics_params = [None, {"min_r": 0, "max_r": 0.25, "n": 20}]

    distance_measures = []
    weights = []
    weight = 1.0 / len(metrics)
    for metric, metric_params in zip(metrics, metrics_params):
        if callable(metric):
            _metrics = metric(**(metric_params) or {})
        elif metric not in _METRICS:
            raise ValueError("metric (%r) is not supported" % metric)

        _metrics = _METRICS[metric](**(metric_params or {}))
        for distance_measure in _metrics:
            distance_measures.append(distance_measure)
            weights.append(weight / len(_metrics))

    return distance_measures, np.array(weights, dtype=np.double)


class ProximityTreeClassifier(TreeClassifierMixin, BaseTree):
    """A proximity tree defines a k-branching tree based on pivot-time series.

    Examples
    --------
    >>> from wildboar.datasets import load_dataset
    >>> from wildboar.tree import ProximityTreeClassifier
    >>> x, y = load_dataset("GunPoint")
    >>> f = ProximityTreeClassifier(n_pivot=10, criterion="gini")
    >>> f.fit(x, y)

    References
    ----------
    Lucas, Benjamin, Ahmed Shifaz, Charlotte Pelletier, Lachlan O’Neill, Nayyar Zaidi,
    Bart Goethals, François Petitjean, and Geoffrey I. Webb. (2019)
        Proximity forest: an effective and scalable distance-based classifier for time
        series. Data Mining and Knowledge Discovery
    """

    @unstable
    def __init__(
        self,
        n_pivot=1,
        *,
        criterion="entropy",
        pivot_sample="label",
        metric_sample="weighted",
        metrics=None,
        metrics_params=None,
        force_dim=None,
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

        metrics : str or list, optional
            The distance metrics

        metrics_params : list, optional
            The params to the metrics

        max_depth : int, optional
            The maximum tree depth.

        min_samples_split : int, optional
            The minimum number of samples to consider a split.

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf.

        min_impurity_decrease : float, optional
            The minimum impurity decrease to build a sub-tree.

        class_weight : array-like of shape (n_labels, ) or "balanced", optional
            The class weights.

        random_state : int or RandomState, optional
            The pseudo random number generator.
        """
        super().__init__(
            force_dim=force_dim,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        self.n_pivot = n_pivot
        self.criterion = criterion
        self.pivot_sample = pivot_sample
        self.metric_sample = metric_sample
        self.metrics = metrics
        self.metrics_params = metrics_params
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
            metrics=self.metrics, metrics_params=self.metrics_params
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
