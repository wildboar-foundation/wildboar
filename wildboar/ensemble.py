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
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .tree import ShapeletTreeClassifier, ExtraShapeletTreeClassifier, ExtraShapeletTreeRegressor
from .tree import ShapeletTreeRegressor

__all__ = ["ShapeletForestClassifier",
           "ExtraShapeletTreesClassifier",
           "ShapeletForestRegressor",
           "ExtraShapeletTreesRegressor"]


class BaseShapeletForest(BaseEstimator, metaclass=ABCMeta):
    def __init__(self,
                 *,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 n_shapelets=10,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 bootstrap=True,
                 n_jobs=None,
                 random_state=None):
        """A shapelet forest classifier
        """
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params
        self.random_state = random_state
        self.n_timestep = None
        self.n_dims = None

    def _validate_x_predict(self, x, check_input):
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(
                x.ndim))
        if self.n_dims > 1 and x.ndim != 3:
            raise ValueError("illegal input dimensions X.ndim != 3")
        if x.shape[-1] != self.n_timestep:
            raise ValueError("illegal input shape ({} != {})".format(
                x.shape[-1], self.n_timestep))
        if x.ndim > 2 and x.shape[1] != self.n_dims:
            raise ValueError("illegal input shape ({} != {}".format(
                x.shape[1], self.n_dims))
        if check_input:
            x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.dtype != np.float64 or not x.flags.contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)
        x = x.reshape(x.shape[0], self.n_dims * self.n_timestep)
        return x

    @abstractmethod
    def _make_estimator(self, random_state):
        pass

    @property
    def estimators_(self):
        return self.ensemble_.estimators_


class ShapeletForestClassifier(ClassifierMixin, BaseShapeletForest):
    def __init__(self,
                 *,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 n_shapelets=10,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 bootstrap=True,
                 n_jobs=None,
                 random_state=None):
        """A shapelet forest classifier"""
        super(ShapeletForestClassifier, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_shapelets=n_shapelets,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state
        )
        self.n_classes_ = None

    def predict(self, X, check_input=True):
        return self.classes_[np.argmax(
            self.predict_proba(X, check_input=check_input), axis=1)]

    def predict_proba(self, x, check_input=True):
        check_is_fitted(self, ["ensemble_"])
        x = self._validate_x_predict(x, check_input)
        return self.ensemble_.predict_proba(x)

    def predict_log_proba(self, x, check_input=True):
        check_is_fitted(self, ["ensemble_"])
        x = self._validate_x_predict(x, check_input)
        return self.ensemble_.predict_log_proba(x)

    def fit(self, x, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest classifier
        """
        random_state = check_random_state(self.random_state)
        if check_input:
            x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, ensure_2d=False)

        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = x.shape[0]
        self.n_timestep = x.shape[-1]
        if x.ndim > 2:
            n_dims = x.shape[1]
        else:
            n_dims = 1

        self.n_dims = n_dims

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if x.dtype != np.float64 or not x.flags.contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        shapelet_tree_classifier = self._make_estimator(random_state)

        if n_dims > 1:
            shapelet_tree_classifier.force_dim = n_dims

        self.ensemble_ = BaggingClassifier(
            base_estimator=shapelet_tree_classifier,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        x = x.reshape(n_samples, n_dims * self.n_timestep)
        self.ensemble_.fit(x, y, sample_weight=sample_weight)
        return self

    def _make_estimator(self, random_state):
        return ShapeletTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_shapelets=self.n_shapelets,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=random_state,
        )


class ExtraShapeletTreesClassifier(ShapeletForestClassifier):
    def _make_estimator(self, random_state):
        return ExtraShapeletTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=random_state,
        )


class ShapeletForestRegressor(RegressorMixin, BaseShapeletForest):
    def __init__(self,
                 *,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 n_shapelets=10,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 bootstrap=True,
                 n_jobs=None,
                 random_state=None):
        """A shapelet forest regressor"""
        super(ShapeletForestRegressor, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_shapelets=n_shapelets,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state
        )

    def predict(self, x, check_input=True):
        check_is_fitted(self, ["ensemble_"])
        x = self._validate_x_predict(x, check_input)
        return self.ensemble_.predict(x)

    def fit(self, x, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest regressor
        """
        random_state = check_random_state(self.random_state)
        if check_input:
            x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, dtype=np.float64, ensure_2d=False, order="C")

        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = x.shape[0]
        self.n_timestep = x.shape[-1]
        if x.ndim > 2:
            n_dims = x.shape[1]
        else:
            n_dims = 1

        self.n_dims = n_dims

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if x.dtype != np.float64 or not x.flags.contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)

        if y.dtype != np.float64 or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.float64)

        shapelet_tree_regressor = self._make_estimator(random_state)

        if n_dims > 1:
            shapelet_tree_regressor.force_dim = n_dims

        self.ensemble_ = BaggingRegressor(
            base_estimator=shapelet_tree_regressor,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        x = x.reshape(n_samples, n_dims * self.n_timestep)
        self.ensemble_.fit(x, y, sample_weight=sample_weight)
        return self

    def _make_estimator(self, random_state):
        return ShapeletTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_shapelets=self.n_shapelets,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=random_state,
        )


class ExtraShapeletTreesRegressor(ShapeletForestRegressor):
    def _make_estimator(self, random_state):
        return ExtraShapeletTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=random_state,
        )
