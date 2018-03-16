import math
import numpy as np

from pypf.shapelet import sliding_distance, z_norm
from sklearn.base import ClassifierMixin


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
        start = self._random_state.randint(x.shape[1] - 2)
        end = self._random_state.randint(start, x.shape[1])
        idx = self._random_state.randint(x.shape[1])
        print(x[idx, start:end])

    def find_threshold(self, x, y, n_classes):
        best_shapelet = None
        best_threshold = math.inf
        best_impurity = math.inf
        best_left = None
        best_right = None
        for i in range(self._n_shapelets):
            shapelet = self.draw_shapelet(x)

    def build_tree(self, x, y, n_classes):
        if x.shape[0] < 2 or n_classes < 2:
            distribution = np.bincount(y, minlength=n_classes) / len(y)
            return Leaf(distribution)

        # left indices smaller than threshold
        # right indices larger than threshold
        left, right, shapelet, threshold = self.find_threshold(x, y, n_classes)
        left_node = build_tree(x[left, :], y[left], n_classes)
        right_node = build_tree(x[right, :], y[right, :], n_classes)
        return Branch(left_node, right_node, shapelet, threshold)



class PfTree(ClassifierMixin):
    def __init__(self,
                 max_depth=None,
                 min_samples_leaf=2,
                 n_shapelets=10,
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state if random_state else np.random.RandomState()
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
