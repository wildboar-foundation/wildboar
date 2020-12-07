# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten
from copy import deepcopy

import numpy as np
from wildboar.distance import distance, matches

from wildboar.ensemble._ensemble import BaseShapeletForestClassifier
from wildboar.explain.counterfactual.base import BaseCounterfactual

from sklearn.utils.validation import check_is_fitted


def _shapelet_transform(shapelet, x, start_index, theta):
    x_shapelet = x[start_index:(start_index + shapelet.shape[0])]
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
    This implementation only supports the reversible algorithm described by Karlsson (2020)

    References
    ----------
    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2020).
        Locally and globally explainable time series tweaking.
        Knowledge and Information Systems, 62(5), 1671-1700.

    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2018).
        Explainable time series tweaking via irreversible and reversible temporal
        transformations. In 2018 IEEE International Conference on Data Mining (ICDM)
    """

    def __init__(self, *, epsilon=1, batch_size=1, random_state=10):
        """

        Parameters
        ----------
        epsilon : float, optional
            Control the degree of change from the decision threshold

        batch_size : int, optional
            Batch size when evaluating the cost and predictions of counterfactual candidates

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
        counterfactuals = np.empty(x.shape)
        success = np.zeros(x.shape[0], dtype=bool)
        for i in range(x.shape[0]):
            t = self._transform_single(x[i, :], y[i])
            if t is not None:
                counterfactuals[i, :] = t
                success[i] = True
            else:
                success[i] = False

        return counterfactuals, success

    def _transform_single(self, x, y):
        path_list = self.paths_[y]
        x_prime = np.empty([len(path_list), x.shape[0]])
        for i, path in enumerate(path_list):
            x_i_prime = self._transform_single_path(x.copy(), path)
            x_prime[i, :] = x_i_prime

        cost = _compute_cost(x_prime, x)
        cost_sort = np.argsort(cost)

        # If the cost of prediction didn't carry the overhead of
        # copying data to different cores due to the python
        # implementation of `BaggingClassifier` a `batch_size` of 1
        # would be optimal. However, empirically half of the
        # conversions seems to be the fastest in practice for this
        # implementation.
        #
        # Note that the cost is ordered in increasing order; hence, if
        # a conversion is successful there can exist no other
        # successful transformation with lower cost.
        batch_size = round(x_prime.shape[0] * self.batch_size) + 1
        for i in range(0, len(cost_sort), batch_size):
            end = min(len(cost_sort), i + batch_size)
            cost_sort_i = cost_sort[i:end]
            x_prime_i = x_prime[cost_sort_i, :]
            y_prime_i = self._estimator.predict(x_prime_i)
            condition_i = y_prime_i == y
            x_prime_i = x_prime_i[condition_i]
            y_prime_i = y_prime_i[condition_i]
            if x_prime_i.shape[0] > 0:
                print(cost[cost_sort_i[condition_i]][:2], y_prime_i[0:2])
                return x_prime_i[0, :]

        return None

    def _transform_single_path(self, x, path):
        for direction, (dim, shapelet), threshold in path:
            if direction == "<=":
                dist, location = distance(shapelet, x, metric="euclidean", return_index=True)
                if dist > threshold:
                    impute_shape = _shapelet_transform(shapelet, x, location, threshold - self.epsilon)
                    x[location:(location + len(shapelet))] = impute_shape
            else:
                dist, location = distance(shapelet, x, metric="euclidean", return_index=True)
                while dist - threshold < 0.0001:
                    impute_shape = _shapelet_transform(shapelet, x, location, threshold + self.epsilon)
                    x[location:(location + len(shapelet))] = impute_shape
                    dist, location = distance(shapelet, x, metric="euclidean", return_index=True)
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
