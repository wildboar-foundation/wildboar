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

import numpy as np

from wildboar.tree import ShapeletTreeClassifier
from wildboar.tree import ShapeletTreeRegressor

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

from sklearn.utils import check_random_state
from sklearn.utils import check_array


class ShapeletForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
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

    def predict(self, X, check_input=True):
        return self.classes_[np.argmax(
            self.predict_proba(X, check_input=check_input), axis=1)]

    def predict_proba(self, X, check_input=True):
        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(
                X.ndim))

        if self.n_dims_ > 1 and X.ndim != 3:
            raise ValueError("illegal input dimensions X.ndim != 3")

        if X.shape[-1] != self.n_timestep_:
            raise ValueError("illegal input shape ({} != {})".format(
                X.shape[-1], self.n_timestep_))

        if X.ndim > 2 and X.shape[1] != self.n_dims_:
            raise ValueError("illegal input shape ({} != {}".format(
                X.shape[1], self.n_dims))

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        X = X.reshape(X.shape[0], self.n_dims_ * self.n_timestep_)
        return self.bagging_classifier_.predict_proba(X)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest classifier
        """
        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, ensure_2d=False)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = X.shape[0]
        self.n_timestep_ = X.shape[-1]
        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        self.n_dims_ = n_dims

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

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        shapelet_tree_classifier = ShapeletTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_shapelets=self.n_shapelets,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=random_state,
        )

        if n_dims > 1:
            shapelet_tree_classifier.force_dim = n_dims

        self.bagging_classifier_ = BaggingClassifier(
            base_estimator=shapelet_tree_classifier,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        X = X.reshape(n_samples, n_dims * self.n_timestep_)
        self.bagging_classifier_.fit(X, y, sample_weight=sample_weight)
        return self


class ShapeletForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
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
        """A shapelet forest regressor
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

    def predict(self, X, check_input=True):
        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(
                X.ndim))

        if self.n_dims_ > 1 and X.ndim != 3:
            raise ValueError("illegal input dimensions X.ndim != 3")

        if X.shape[-1] != self.n_timestep_:
            raise ValueError("illegal input shape ({} != {})".format(
                X.shape[-1], self.n_timestep_))

        if X.ndim > 2 and X.shape[1] != self.n_dims_:
            raise ValueError("illegal input shape ({} != {}".format(
                X.shape[1], self.n_dims))

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        X = X.reshape(X.shape[0], self.n_dims_ * self.n_timestep_)
        return self.bagging_regressor_.predict(X)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest regressor
        """
        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, dtype=np.float64, ensure_2d=False, order="C")

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = X.shape[0]
        self.n_timestep_ = X.shape[-1]
        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        self.n_dims_ = n_dims

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if y.dtype != np.float64 or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.float64)

        shapelet_tree_regressor = ShapeletTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_shapelets=self.n_shapelets,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=random_state,
        )

        if n_dims > 1:
            shapelet_tree_regressor.force_dim = n_dims

        self.bagging_regressor_ = BaggingRegressor(
            base_estimator=shapelet_tree_regressor,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        X = X.reshape(n_samples, n_dims * self.n_timestep_)
        self.bagging_regressor_.fit(X, y, sample_weight=sample_weight)
        return self
