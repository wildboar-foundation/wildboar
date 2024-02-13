# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers
from copy import deepcopy

import numpy as np
from sklearn.utils import check_random_state, resample
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ...base import BaseEstimator, CounterfactualMixin, ExplainerMixin
from ...distance._distance import (
    _METRICS,
    paired_distance,
    pairwise_subsequence_distance,
)
from ...ensemble._ensemble import BaseShapeletForestClassifier


def _min_euclidean_distance(shapelet, x):
    return pairwise_subsequence_distance(
        shapelet.reshape(-1),
        x.reshape(-1),
        dim=0,
        metric="euclidean",
        return_index=True,
    )


def _shapelet_transform(shapelet, x, start_index, theta):
    x_shapelet = x[start_index : (start_index + shapelet.shape[0])]
    shapelet_diff = x_shapelet - shapelet
    dist = np.linalg.norm(shapelet_diff)
    if dist == 0:
        # If the distance between the shapelets is zero, we move it in a random
        # direction towards the distance threshold.
        # TODO: ensure that this uses a consistent random_seed for repeatable runs.
        x_shapelet = np.random.uniform(shapelet.shape)
        shapelet_diff = x_shapelet - shapelet
        dist = np.linalg.norm(shapelet_diff)

    return shapelet + shapelet_diff / dist * theta


def _path_transform(x, path, epsilon):
    for direction, (dim, shapelet), threshold in path:
        if x.ndim == 2:
            x_dim = x[dim]
        else:
            x_dim = x

        dist, location = _min_euclidean_distance(shapelet, x_dim)
        if direction < 0:
            if dist > threshold:
                impute_shape = _shapelet_transform(
                    shapelet, x_dim, location, threshold - epsilon
                )
                x_dim[location : location + len(shapelet)] = impute_shape
        else:
            while dist - threshold < 0:
                impute_shape = _shapelet_transform(
                    shapelet, x_dim, location, threshold + epsilon
                )
                x_dim[location : location + len(shapelet)] = impute_shape
                dist, location = _min_euclidean_distance(shapelet, x_dim)
    return x


class PredictionPaths:
    def __init__(self, classes):
        self.classes = classes
        self._paths = {c: [] for c in classes}

    def _append(self, tree):
        left = tree.left
        right = tree.right
        threshold = tree.threshold
        shapelet = tree.attribute
        value = tree.value

        def recurse(node_id, path):
            if left[node_id] != -1:
                left_path = path.copy()
                (dim, (dim_, data)) = shapelet[node_id]
                left_path.append((-1, (dim, data), threshold[node_id]))
                recurse(left[node_id], left_path)

                right_path = path.copy()
                right_path.append((1, (dim, data), threshold[node_id]))
                recurse(right[node_id], right_path)
            else:
                self._paths[self.classes[np.argmax(value[node_id])]].append(path)

        recurse(0, [])

    def prune(self, max_paths, random_state):
        self._paths = {
            label: resample(
                paths,
                replace=False,
                n_samples=math.ceil(len(paths) * max_paths),
                random_state=random_state.randint(np.iinfo(np.int32).max),
            )
            for label, paths in self._paths.items()
        }

    def __contains__(self, item):
        return item in self._paths

    def __getitem__(self, item):
        return self._paths[item]


def _cost_wrapper(metric):
    def f(x, y, aggregation=np.mean):
        if x.ndim == 1 and y.ndim == 1:
            return paired_distance(x.reshape(1, -1), y.reshape(1, -1), metric=metric)
        elif x.ndim == 2 and y.ndim == 2:
            return paired_distance(x, y, metric=metric)
        elif x.ndim == 3 and y.ndim == 3:
            return aggregation(
                [
                    paired_distance(x, y, metric=metric, dim=i)
                    for i in range(x.shape[1])
                ],
                axis=0,
            )
        else:
            raise ValueError("invalid dim")

    return f


_AGGREGATION = {
    "median": np.median,
    "mean": np.mean,
}


class ShapeletForestCounterfactual(CounterfactualMixin, ExplainerMixin, BaseEstimator):
    """
    Counterfactual explanations for shapelet forest classifiers.

    Parameters
    ----------
    cost : {"euclidean", "cosine", "manhattan"} or callable, optional
        The cost function to determine the goodness of counterfactual.
    aggregation : callable, optional
        The aggregation function for the cost of multivariate counterfactuals.
    epsilon : float, optional
        Control the degree of change from the decision threshold.
    batch_size : float, optional
        Batch size when evaluating the cost and predictions of
        counterfactual candidates. The default setting is to evaluate
        all counterfactual samples.

        .. versionchanged:: 1.1
            The default value changed to 0.1
    max_paths : float, optional
        Sample a fraction of the positive prediction paths.

        .. versionadded:: 1.1
            Add support for subsampling prediction paths.
    verbose : bool, optional
        Print information to stdout during execution.
    random_state : RandomState or int, optional
        Pseudo-random number for consistency between different runs.

    Attributes
    ----------
    paths_ : dict
        A dictionary of prediction paths per label

    Warnings
    --------
    Only shapelet forests fit with the Euclidean distance is supported i.e.,
    ``metric="euclidean"``

    Notes
    -----
    This implementation only supports the reversible algorithm
    described by Karlsson (2020)

    References
    ----------
    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2020).
        Locally and globally explainable time series tweaking.
        Knowledge and Information Systems, 62(5), 1671-1700.

    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2018).
        Explainable time series tweaking via irreversible and reversible temporal
        transformations. In 2018 IEEE International Conference on Data Mining (ICDM)
    """

    _parameter_constraints: dict = {
        "cost": [StrOptions(_METRICS.keys()), callable],
        "aggregation": [StrOptions(_AGGREGATION.keys()), callable],
        "epsilon": [Interval(numbers.Real, 0, None, closed="right")],
        "batch_size": [Interval(numbers.Real, 0, 1, closed="right")],
        "max_paths": [Interval(numbers.Real, 0, 1, closed="right")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        cost="euclidean",
        aggregation="mean",
        epsilon=1.0,
        batch_size=0.1,
        max_paths=1.0,
        verbose=False,
        random_state=None,
    ):
        self.cost = cost
        self.aggregation = aggregation
        self.epsilon = epsilon
        self.random_state = random_state
        self.batch_size = batch_size
        self.max_paths = max_paths
        self.verbose = verbose

    def _validate_estimator(self, estimator, allow_3d=False):
        if not isinstance(estimator, BaseShapeletForestClassifier):
            raise ValueError(
                "estimator must be ShapeletForestClassifier, not %r"
                % type(estimator).__qualname__
            )

        return super()._validate_estimator(estimator, allow_3d)

    def fit(self, estimator, x=None, y=None):
        self._validate_params()
        estimator = self._validate_estimator(estimator, allow_3d=True)

        self.cost_ = _cost_wrapper(self.cost)
        if isinstance(self.aggregation, str):
            self.aggregation_ = _AGGREGATION[self.aggregation]
        else:
            self.aggregation_ = self.aggregation

        self.estimator_ = deepcopy(estimator)
        self.paths_ = PredictionPaths(estimator.classes_)

        for base_estimator in self.estimator_.estimators_:
            self.paths_._append(base_estimator.tree_)

        self.epsilon_ = self.epsilon
        self.batch_size_ = self.batch_size
        self.paths_.prune(self.max_paths, check_random_state(self.random_state))
        return self

    def explain(self, x, y):
        check_is_fitted(self)
        x, y = self._validate_data(x, y, allow_3d=True, reset=False, dtype=float)
        counterfactuals = np.empty(x.shape)

        for i in range(x.shape[0]):
            if self.verbose:
                print(
                    f"Generating counterfactual for the {i}:th sample. "
                    f"The desired target label is {y[i]}."
                )

            if y[i] not in self.paths_:
                raise ValueError("unknown label, got %r" % y)

            t = self._candidates(x[i], y[i])
            if t is not None:
                counterfactuals[i] = t
            else:
                counterfactuals[i] = x[i]

        return counterfactuals

    def _candidates(self, x, y):
        prediction_paths = self.paths_[y]
        n_counterfactuals = len(prediction_paths)
        counterfactuals = np.empty((n_counterfactuals,) + x.shape)
        for i, path in enumerate(prediction_paths):
            counterfactuals[i] = _path_transform(x.copy(), path, self.epsilon_)

        # Note that the cost is ordered in increasing order; hence, if a
        # conversion is successful there can exist no other successful
        # transformation with lower cost.
        x = np.broadcast_to(x, counterfactuals.shape)
        cost_sort = np.argsort(
            self.cost_(counterfactuals, x, aggregation=self.aggregation_)
        )

        batch_size = math.ceil(n_counterfactuals * self.batch_size_)
        for i in range(0, n_counterfactuals, batch_size):
            batch_cost = cost_sort[i : min(n_counterfactuals, i + batch_size)]
            batch_counterfactuals = counterfactuals[batch_cost]
            batch_prediction = self.estimator_.predict(batch_counterfactuals)
            batch_counterfactuals = batch_counterfactuals[batch_prediction == y]
            if batch_counterfactuals.shape[0] > 0:
                return batch_counterfactuals[0]

        return None


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
