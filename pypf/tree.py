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
                 unscale_threshold=False,
                 random_state=None):
        self.max_depth = max_depth
        self.scale = scale
        self.unscale_threshold = unscale_threshold
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
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)

        # TODO: this is to crude, the TreeBuilder should be enhanced
        # with the capabilities of sample_weight, i.e., here we only
        # consider if an instance is included, not how many times
        if sample_weight is not None:
            indicies = indicies[sample_weight > 0]

        indicies = np.ascontiguousarray(indicies)
        tree_builder = ShapeletTreeBuilder(
            self.n_shapelets,
            self.scale,
            self.unscale_threshold,
            random_state,
        )
        self.classes_ = np.unique(y)
        tree_builder.init(X, y, len(self.classes_))
        self.tree = tree_builder.build_tree(indicies)

        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = check_array(X, dtype=np.float64)
        X = np.ascontiguousarray(X)
        predictor = ShapeletTreePredictor(X, len(self.classes_))
        proba = predictor.predict_proba(self.tree)
        return proba
