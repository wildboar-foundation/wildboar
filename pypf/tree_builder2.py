import math

import numpy as np

from pypf._sliding_distance import sliding_distance_one
from pypf._sliding_distance import sliding_distance
from pypf._distribution import get_class_distribution
from pypf._impurity import safe_info
#from pypf._partition import partition
from sklearn.utils import check_random_state


def partition(idx, d, y, n_classes):
    if len(idx) == 1:
        print("WHAT!?")
        return 0.0, d[0], idx, np.array([], dtype=int)

    n_examples = len(idx)
    order = np.argsort(d)
    left_d = np.zeros(n_classes, dtype=np.float64)
    right_d = np.zeros(n_classes, dtype=np.float64)
    for i in range(n_examples):
        right_d[y[idx[i]]] += 1

    prev_dist = d[order[0]]
    prev_label = y[idx[order[0]]]

    lt_w = 1
    gt_w = n_examples - 1
    left_d[prev_label] += 1
    right_d[prev_label] -= 1

    entropy = safe_info(lt_w, left_d, gt_w, right_d, n_examples)
    threshold = prev_dist
    threshold_index = 1
    for i in range(1, len(idx)):
        order_i = order[i]
        dist = d[order_i]
        label = y[idx[order_i]]
        same_label = label == prev_label
        if not same_label:
            e = safe_info(lt_w, left_d, gt_w, right_d, n_examples)
            if e < entropy:
                entropy = e
                threshold = (dist + prev_dist) / 2
                threshold_index = i

        prev_label = label
        prev_dist = dist

        lt_w += 1
        gt_w -= 1
        left_d[label] += 1
        right_d[label] -= 1

    return entropy, threshold, threshold_index, order


class Branch:
    def __init__(self, left, right, shapelet, threshold):
        self._left = left
        self._right = right
        self._shapelet = shapelet
        self._threshold = threshold

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def shapelet(self):
        return self._shapelet

    @property
    def threshold(self):
        return self._threshold

    def prnt(self, indent=1):
        print("-" * indent, "branch:")
        print("-" * indent, " shapelet: ", self._shapelet)
        print("-" * indent, " threshold: ", self._threshold)
        print("-" * indent, " left:", end="\n")
        self._left.prnt(indent + 1)
        print("-" * indent, " right:", end="\n")
        self._right.prnt(indent + 1)


class Leaf:
    def __init__(self, proba):
        self._proba = proba

    @property
    def proba(self):
        return self._proba

    def prnt(self, indent):
        print("-" * indent, "leaf: ")
        print("-" * indent, " proba: ", self.proba)


class TreeBuilder:
    def __init__(self, n_shapelets, random_state):
        self._n_shapelets = n_shapelets
        self._random_state = check_random_state(random_state)

    def draw_shapelet(self, idx, x):
        length = self._random_state.randint(3, x.shape[1])
        start = self._random_state.randint(0, x.shape[1] - length)
        i = self._random_state.randint(len(idx))
        shapelet = x[idx[i], start:(start + length)]

        std = np.std(shapelet)
        if std:
            z_norm_shapelet = (shapelet - np.mean(shapelet)) / std
            return z_norm_shapelet, np.argsort(-shapelet)
        else:
            return np.zeros(length), np.arange(length)

    def find_threshold(self, idx, x, y, n_classes):
        best_shapelet = None
        best_threshold = math.inf
        best_impurity = math.inf
        best_order = None
        best_threshold_index = None
        distances = np.empty(len(idx))
        for i in range(self._n_shapelets):
            shapelet, o = self.draw_shapelet(idx, x)
            sliding_distance(shapelet, x, idx, out=distances)

            impurity, threshold, threshold_index, order = partition(
                idx, distances, y, n_classes)

            if impurity < best_impurity:
                best_impurity = impurity
                best_threshold = threshold
                best_threshold_index = threshold_index
                best_order = order
                best_shapelet = shapelet

        left = idx[best_order[:best_threshold_index]]
        right = idx[best_order[best_threshold_index:]]
        return left, right, best_shapelet, best_threshold

    def build_tree(self, idx, x, y, n_classes):
        distribution = np.asarray(get_class_distribution(idx, y, n_classes))
        if len(idx) < 2 or np.sum(distribution > 0) < 2:
            return Leaf(distribution)

        # left indices smaller than threshold
        # right indices larger than threshold
        # print("finding threshold")
        left, right, shapelet, threshold = self.find_threshold(
            idx, x, y, n_classes)
        # print("found threshold for {}".format(idx.shape))
        #        print(shapelet, threshold, left, right)
        if len(left) > 0 and len(right) > 0:
            left_node = self.build_tree(left, x, y, n_classes)
            right_node = self.build_tree(right, x, y, n_classes)
            return Branch(left_node, right_node, shapelet, threshold)
        else:
            return Leaf(distribution)
