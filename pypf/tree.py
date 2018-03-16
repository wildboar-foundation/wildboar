import math
import numpy as np

from pypf._sliding_distance import sliding_distance
from sklearn.base import ClassifierMixin
from scipy.special import entr


def partition(d, y, n_classes):
    order = np.argsort(d)
    d = d[order]
    y = y[order]

    left_d = np.zeros(n_classes)
    right_d = np.bincount(y, minlength=n_classes) / len(y)

    prev_dist = d[0]
    prev_label = y[0]

    weight = 1.0 / len(y)
    lt_w = weight
    gt_w = 1 - lt_w
    left_d[prev_label] += weight
    right_d[prev_label] -= weight

    entropy = lt_w * np.sum(entr(left_d)) + gt_w * np.sum(entr(right_d))
    threshold = prev_dist
    for i in range(1, len(d)):
        dist = d[i]
        label = y[i]

        lt_w += weight
        gt_w -= weight
        left_d[label] += weight
        right_d[label] -= weight

        same_label = label == prev_label
        if not same_label:
            e = lt_w * np.sum(entr(left_d)) + gt_w * np.sum(entr(right_d))
            if e < entropy:
                print(lt_w, np.sum(entr(left_d)), gt_w, np.sum(entr(right_d)))
                entropy = e
                threshold = (dist + prev_dist) / 2

        prev_label = label
        prev_dist = dist
    print(list(d))
    print(list(y))
    print(entropy, threshold)

    return entropy, threshold, [], []


class Branch:
    def __init__(self, left, right, shapelet, threshold):
        self._left = left
        self._right = right
        self._shapelet = shapelet
        self._threshold = threshold


class Leaf:
    def __init__(self, proba):
        self.proba = proba


class TreeBuilder:
    def __init__(self, n_shapelets, random_state):
        self._n_shapelets = n_shapelets
        self._random_state = random_state

    def draw_shapelet(self, x):
        length = self._random_state.randint(2, x.shape[1] - 2)
        start = self._random_state.randint(0, x.shape[1] - length)
        idx = self._random_state.randint(x.shape[1])
        s = x[idx, start:(start + length)]
        return (s - np.mean(s)) / np.std(s), np.argsort(-s)

    def find_threshold(self, x, y, n_classes):
        best_shapelet = None
        best_threshold = math.inf
        best_impurity = math.inf
        best_left = None
        best_right = None
        for i in range(self._n_shapelets):
            distances = np.empty(x.shape[0])
            s, o = self.draw_shapelet(x)
            for j in range(x.shape[0]):
                distances[j] = sliding_distance(s, x[j, :])

            impurity, threshold, left, right = partition(
                distances, y, n_classes)
        return [], [], best_shapelet, best_threshold

    def build_tree(self, x, y, n_classes):
        if x.shape[0] < 2 or n_classes < 2:
            distribution = np.bincount(y, minlength=n_classes) / len(y)
            return Leaf(distribution)

        # left indices smaller than threshold
        # right indices larger than threshold
        left, right, shapelet, threshold = self.find_threshold(x, y, n_classes)
        if left and right:
            left_node = self.build_tree(x[left, :], y[left], n_classes)
            right_node = self.build_tree(x[right, :], y[right, :], n_classes)
            return Branch(left_node, right_node, shapelet, threshold)
        else:
            distribution = np.bincount(y, minlength=n_classes) / len(y)
            return Leaf(distribution)


class PfTree(ClassifierMixin):
    def __init__(self,
                 max_depth=None,
                 min_samples_leaf=2,
                 n_shapelets=10,
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state if random_state else np.random.RandomState(
        )
        self.n_shapelets = n_shapelets

    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError("shape 0 of x and y must match {} != {}".format(
                x.shape[0], y.shape[0]))
        tree_builder = TreeBuilder(self.n_shapelets, self.random_state)
        n_classes = len(np.unique(y))
        return tree_builder.build_tree(x, y, n_classes)

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass
