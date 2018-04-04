import numpy as np

from pypf._tree_builder import ShapeletTreeBuilder
from pypf._tree_builder import ShapeletTreePredictor

from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils import check_array


class PfTree(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 max_depth=None,
                 min_samples_leaf=2,
                 n_shapelets=10,
                 scale=True,
                 random_state=None):
        self.max_depth = max_depth
        self.scale = scale
        self.max_depth = max_depth or 2**31
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

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)
        tree_builder = ShapeletTreeBuilder(
            self.n_shapelets,
            self.max_depth,
            self.scale,
            random_state,
        )

        self.classes_ = np.unique(y)
        tree_builder.init(X, y, len(self.classes_), sample_weight)
        self.tree = tree_builder.build_tree()

        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = check_array(X, dtype=np.float64)
        X = np.ascontiguousarray(X)
        predictor = ShapeletTreePredictor(X, len(self.classes_))
        proba = predictor.predict_proba(self.tree)
        return proba
