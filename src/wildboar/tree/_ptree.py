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

from wildboar.distance import _DISTANCE_MEASURE
from wildboar.tree._cptree import (
    EntropyCriterion,
    GiniCriterion,
    LabelPivotSampler,
    Tree,
    TreeBuilder,
    UniformDistanceMeasureSampler,
)
from wildboar.tree.base import BaseTree, TreeClassifierMixin
from wildboar.utils.data import check_dataset

_CLF_CRITERION = {
    "gini": GiniCriterion,
    "entropy": EntropyCriterion,
}
_PIVOT_SAMPLER = {"label": LabelPivotSampler}
_DISTANCE_MEASURE_SAMPLER = {"uniform": UniformDistanceMeasureSampler}


class ProximityTreeClassifier(TreeClassifierMixin, BaseTree):
    def __init__(
        self,
        n_pivot=1,
        *,
        criterion="entropy",
        pivot_sample="label",
        metric_sample="uniform",
        force_dim=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        class_weight=None,
        random_state=None,
    ):
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
            raise ValueError()

        if self.pivot_sample not in _PIVOT_SAMPLER:
            raise ValueError()

        if self.metric_sample not in _DISTANCE_MEASURE_SAMPLER:
            raise ValueError()

        criterion = _CLF_CRITERION[self.criterion](y, self.n_classes_)
        pivot_sampler = _PIVOT_SAMPLER[self.pivot_sample]()
        distance_measure_sampler = _DISTANCE_MEASURE_SAMPLER[self.metric_sample]()
        tree = Tree(
            [
                _DISTANCE_MEASURE["dtw"](r=0.2),
                _DISTANCE_MEASURE["euclidean"](),
            ],
            self.n_classes_,
        )
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
        self.tree_ = tree_builder.tree_
