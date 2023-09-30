import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator


class DiffTransform(TransformerMixin, BaseEstimator):
    def __init__(self, order=1):
        self.order = order

    def fit(self, X, y=None):
        self._validate_data(X)
        self.order_ = self.order
        return self

    def transform(self, X):
        self._validate_data(X, reset=False)
        check_is_fitted(self)
        return np.diff(X, n=self.order_, axis=-1)
