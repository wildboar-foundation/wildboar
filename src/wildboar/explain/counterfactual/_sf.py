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
import math
from copy import deepcopy
from functools import partial

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

from wildboar.distance import distance
from wildboar.ensemble._ensemble import BaseShapeletForestClassifier
from wildboar.explain.counterfactual.base import BaseCounterfactual

MIN_MATCHING_DISTANCE = 0.0001

euclidean_distance = partial(distance, metric="euclidean", return_index=True)


def _shapelet_transform(shapelet, x, start_index, theta):
    x_shapelet = x[start_index : (start_index + shapelet.shape[0])]
    shapelet_diff = x_shapelet - shapelet
    dist = np.linalg.norm(shapelet_diff)
    if dist == 0:
        x_shapelet = np.random.uniform(shapelet.shape)
        shapelet_diff = x_shapelet - shapelet
        dist = np.linalg.norm(shapelet_diff)

    return shapelet + shapelet_diff / dist * theta


class PredictionPaths:
    def __init__(self, classes):
        self.classes = classes
        self._paths = {c: [] for c in classes}

    def _append(self, tree):
        left = tree.left
        right = tree.right
        threshold = tree.threshold
        shapelet = tree.shapelet
        value = tree.value

        def recurse(node_id, path):
            if left[node_id] != -1:
                left_path = path.copy()
                (dim, data) = shapelet[node_id]
                left_path.append((-1, (dim - 1, data), threshold[node_id]))
                recurse(left[node_id], left_path)

                right_path = path.copy()
                right_path.append((1, (dim - 1, data), threshold[node_id]))
                recurse(right[node_id], right_path)
            else:
                self._paths[self.classes[np.argmax(value[node_id])]].append(path)

        recurse(0, [])

    def __contains__(self, item):
        return item in self._paths

    def __getitem__(self, item):
        return self._paths[item]


def _compute_cost(a, b):
    if a.ndim == 1 and b.ndim == 1:
        return np.linalg.norm(a - b)
    else:
        return np.linalg.norm(a - b, axis=1)


class ShapeletForestCounterfactual(BaseCounterfactual):
    """Counterfactual explanations for shapelet forest classifiers

    Attributes
    ----------
    paths_ : dict
        A dictionary of prediction paths per label

    Notes
    -----
    This implementation only supports the reversible algorithm
    described by Karlsson (2020)

    Warnings
    --------
    Only shapelet forests fit with the Euclidean distance is supported i.e.,
    ``metric="euclidean"``

    References
    ----------
    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2020).
        Locally and globally explainable time series tweaking.
        Knowledge and Information Systems, 62(5), 1671-1700.

    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2018).
        Explainable time series tweaking via irreversible and reversible temporal
        transformations. In 2018 IEEE International Conference on Data Mining (ICDM)
    """

    def __init__(self, *, epsilon=1.0, batch_size=1, random_state=10):
        """

        Parameters
        ----------
        epsilon : float, optional
            Control the degree of change from the decision threshold

        batch_size : float, optional
            Batch size when evaluating the cost and predictions of
            counterfactual candidates. The default setting is to evaluate
            all counterfactual samples.

        random_state : RandomState or int, optional
            Pseudo-random number for consistency between different runs
        """
        self.epsilon = epsilon
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, estimator):
        if not isinstance(estimator, BaseShapeletForestClassifier):
            raise ValueError("unsupported estimator, got %r" % estimator)
        check_is_fitted(estimator)
        self._estimator = deepcopy(estimator)
        self.paths_ = PredictionPaths(estimator.classes_)
        for base_estimator in self._estimator.estimators_:
            self.paths_._append(base_estimator.tree_)
        return self

    def transform(self, x, y):
        check_is_fitted(self, "paths_")
        y = check_array(y, ensure_2d=False)
        x = check_array(x)
        if len(y) != x.shape[0]:
            raise ValueError(
                "Number of labels={} does not match "
                "number of samples={}".format(len(y), x.shape[0])
            )
        counterfactuals = np.empty(x.shape)
        success = np.zeros(x.shape[0], dtype=bool)
        for i in range(x.shape[0]):
            t = self.candidates(x[i, :], y[i])
            if t is not None:
                counterfactuals[i, :] = t
                success[i] = True
            else:
                success[i] = False

        return counterfactuals, success

    def candidates(self, x, y):
        if x.ndim == 2 and x.shape[0] > 1:
            raise ValueError(
                "can only compute candidates for 1 sample, got %d" % x.shape[0]
            )
        # elif x.ndim == 1:
        #     x = x.reshape(1, -1)

        if self.epsilon < 0.0:
            raise ValueError("epsilon must be larger than 0, got %r" % self.epsilon)

        if y not in self.paths_:
            raise ValueError("unknown label, got %r" % y)

        prediction_paths = self.paths_[y]
        n_counterfactuals = len(prediction_paths)

        if isinstance(self.batch_size, float):
            if not 0.0 < self.batch_size <= 1:
                raise ValueError(
                    "batch_size should be in range (0, 1], got %r" % self.batch_size
                )
            batch_size = math.ceil(n_counterfactuals * self.batch_size)
        elif isinstance(self.batch_size, int):
            batch_size = max(0, min(self.batch_size, n_counterfactuals))
        else:
            raise ValueError("unsupported batch_size, got %r" % self.batch_size)

        counterfactuals = np.empty([n_counterfactuals, x.shape[0]])
        for i, path in enumerate(prediction_paths):
            counterfactuals[i, :] = self._path_transform(x.copy(), path)

        # Note that the cost is ordered in increasing order; hence, if a
        # conversion is successful there can exist no other successful
        # transformation with lower cost.
        cost_sort = np.argsort(_compute_cost(counterfactuals, x))
        for i in range(0, n_counterfactuals, batch_size):
            batch_cost = cost_sort[i : min(n_counterfactuals, i + batch_size)]
            batch_counterfactuals = counterfactuals[batch_cost, :]
            batch_prediction = self._estimator.predict(batch_counterfactuals)
            batch_counterfactuals = batch_counterfactuals[batch_prediction == y]
            if batch_counterfactuals.shape[0] > 0:
                return batch_counterfactuals[0, :]

        return None

    def _path_transform(self, x, path):
        for direction, (dim, shapelet), threshold in path:
            dist, location = euclidean_distance(shapelet, x, dim=dim)
            if direction < 0:
                if dist > threshold:
                    impute_shape = _shapelet_transform(
                        shapelet, x, location, threshold - self.epsilon
                    )
                    x[location : location + len(shapelet)] = impute_shape
            elif direction > 0:
                while dist - threshold < 0:
                    impute_shape = _shapelet_transform(
                        shapelet, x, location, threshold + self.epsilon
                    )
                    x[location : location + len(shapelet)] = impute_shape
                    dist, location = euclidean_distance(shapelet, x, dim=dim)
            else:
                raise ValueError("invalid path")
        return x


# class IncrementalTreeLabelTransform(CounterfactualTransformer):
#     def _transform_single_path(self, x, path):
#         for direction, shapelet, threshold in path:
#             if direction == "<=":
#                 dist, location = distance(
#                     shapelet, x, metric="euclidean", return_index=True)
#                 if dist > threshold:
#                     impute_shape = _shapelet_transform(shapelet, x, location,
#                                                        threshold - self.epsilon)
#                     x[location:(location + len(shapelet))] = impute_shape
#             else:
#                 dist, location = distance(
#                     shapelet, x, metric="euclidean", return_index=True)
#                 while dist - threshold < 0.0001:
#                     impute_shape = _shapelet_transform(shapelet, x, location,
#                                                        threshold + self.epsilon)
#                     x[location:(location + len(shapelet))] = impute_shape
#                     dist, location = distance(
#                         shapelet, x, metric="euclidean", return_index=True)
#         return x
#
#
# def locked_iter(locked, start, end):
#     """iterates over unlocked regions. `locked must be sorterd`
#     :param locked: a list of tuples `[(start_lock, end_lock)]`
#     :param start: first index
#     :param end: last index
#     :yield: unlocked regions
#     """
#     if len(locked) == 0:
#         yield start, end
#     for i in range(0, len(locked)):
#         s, e = locked[i]
#         if i == 0:
#             start = 0
#         if s - start > 0:
#             yield start, (start + (s - start))
#         start = e
#
#         if i == len(locked) - 1 and end - start > 0:
#             yield start, end
#
#
# def in_range(i, start, end):
#     return start < i <= end
#
#
# def is_locked(start, end, locked):
#     """Return true if location is locked
#     :param pos: the position
#     :param locked: list of locked positions
#     :returns: true if locked
#     """
#     for s, e in locked:
#         if in_range(s, start, end) or in_range(e, start, end):
#             return True
#     return False
#
#
# class LockingIncrementalTreeLabelTransform(IncrementalTreeLabelTransform):
#     def _transform_single(self, x_orig):
#         path_list = self.paths_[self.to_label_]
#         best_cost = np.inf
#         best_x_prime = np.nan
#         batch_size = round(len(path_list) * self.batch_size) + 1
#         x_prime = np.empty((batch_size, x_orig.shape[0]))
#         n_paths = len(path_list)
#         i = 0
#         prune = 0
#         predictions = 0
#         while i < n_paths:
#             j = 0
#             while j < batch_size and i + j < n_paths:
#                 path = path_list[i + j]
#                 x_prime_j = self._transform_single_path(
#                     x_orig, path, best_cost)
#                 if x_prime_j is not None:
#                     x_prime[j, :] = x_prime_j
#                     j += 1
#                 else:
#                     prune += 1
#                     i += 1
#             i += j
#             if j > 0:
#                 predictions += j
#                 x_prime_pred = x_prime[:j, :]
#                 cond = self.ensemble_.predict(x_prime_pred) == self.to_label_
#                 x_prime_i = x_prime_pred[cond]
#                 if x_prime_i.shape[0] > 0:
#                     cost = self._compute_cost(x_prime_i, x_orig)
#                     argmin_cost = np.argmin(cost)
#                     min_cost = cost[argmin_cost]
#                     if min_cost < best_cost:
#                         best_cost = min_cost
#                         best_x_prime = x_prime_i[argmin_cost]
#
#         return (best_x_prime, predictions / float(len(path_list)),
#                 prune / float(len(path_list)))
#
#     def _transform_single_path(self, x_orig, path, best_cost=np.inf):
#         x_prime = x_orig.copy()
#         locked = []
#         for direction, shapelet, threshold in path:
#             if direction == "<=":
#                 dist, location = distance(
#                     shapelet, x_prime, metric="euclidean", return_index=True)
#
#                 if dist > threshold and not is_locked(
#                         location, location + len(shapelet), locked):
#                     impute_shape = _shapelet_transform(
#                         shapelet, x_prime, location, threshold - self.epsilon)
#
#                     x_prime[location:(location + len(shapelet))] = impute_shape
#                     locked.append((location, location + len(shapelet)))
#
#                     cost = self._compute_cost(x_prime, x_orig)
#                     if cost >= best_cost:
#                         return None
#
#             else:
#                 dist, location = distance(
#                     shapelet, x_prime, metric="euclidean", return_index=True)
#                 while (dist - threshold < 0.0001 and not is_locked(
#                         location, location + len(shapelet), locked)):
#                     impute_shape = _shapelet_transform(
#                         shapelet, x_prime, location, threshold + self.epsilon)
#
#                     x_prime[location:(location + len(shapelet))] = impute_shape
#                     locked.append((location, location + len(shapelet)))
#
#                     cost = self._compute_cost(x_prime, x_orig)
#                     if cost >= best_cost:
#                         return None
#
#                     dist, location = distance(
#                         shapelet,
#                         x_prime,
#                         metric="euclidean",
#                         return_index=True)
#         return x_prime
