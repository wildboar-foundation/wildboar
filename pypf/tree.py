import numpy as np

from pypf._sliding_distance import sliding_distance_one

# from pypf.tree_builder2 import TreeBuilder, Leaf, Branch
from pypf._tree_builder import TreeBuilder, Leaf
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_random_state, check_array


class PfTree(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 max_depth=None,
                 min_samples_leaf=2,
                 n_shapelets=10,
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = check_random_state(random_state)
        self.n_shapelets = n_shapelets

    def fit(self, X, y, sample_weight=None, check_input=True):
        random_state = check_random_state(self.random_state)

        if check_input:
            X = check_array(X, dtype=np.float64)
            y = check_array(y, ensure_2d=False, dtype=np.intp)

        n_samples, n_timesteps = X.shape

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))

        indicies = np.arange(y.shape[0])

        # TODO: this is to crude, the TreeBuilder should be enhanced
        # with the capabilities of sample_weight, i.e., here we only
        # consider if an instance is included, not how many times
        if sample_weight is not None:
            indicies = indicies[sample_weight > 0]

        tree_builder = TreeBuilder(self.n_shapelets, random_state)
        self.classes_ = np.unique(y)

        self.tree = tree_builder.build_tree(indicies, X, y, len(self.classes_))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = check_array(X, dtype=np.float64)

        # TODO: the prediction function should be implemented on top
        # of the TreeBuilder
        output = np.empty([X.shape[0], len(self.classes_)])
        for i in range(X.shape[0]):
            node = self.tree
            while not isinstance(node, Leaf):
                shapelet = node.shapelet
                threshold = node.threshold
                if sliding_distance_one(shapelet, X, i) <= threshold:
                    node = node.left
                else:
                    node = node.right
            output[i, :] = node.proba
        return output
