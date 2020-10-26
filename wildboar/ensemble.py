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
import numbers

import numpy as np
from sklearn.base import OutlierMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble._bagging import BaseBagging
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted

from .tree import ShapeletTreeClassifier, ExtraShapeletTreeClassifier, ExtraShapeletTreeRegressor
from .tree import ShapeletTreeRegressor

__all__ = ["ShapeletForestClassifier",
           "ExtraShapeletTreesClassifier",
           "ShapeletForestRegressor",
           "ExtraShapeletTreesRegressor",
           "IsolationShapeletForest"]


class ShapeletForestMixin:
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


class BaseShapeletForestClassifier(ShapeletForestMixin, BaggingClassifier):
    def __init__(self,
                 base_estimator,
                 *,
                 estimator_params=tuple(),
                 oob_score=False,
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
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_jobs=n_jobs,
            random_state=random_state,
            oob_score=oob_score
        )
        self.estimator_params = estimator_params
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params

    def _parallel_args(self):
        return _joblib_parallel_args(prefer='threads')

    def predict(self, X, check_input=True):
        x = self._validate_x_predict(X, check_input)
        return super().predict(X)

    def predict_proba(self, x, check_input=True):
        x = self._validate_x_predict(x, check_input)
        return super().predict_proba(x)

    def predict_log_proba(self, x, check_input=True):
        x = self._validate_x_predict(x, check_input)
        return super().predict_log_proba(x)

    def fit(self, x, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest classifier
        """
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

        if x.dtype != np.float64 or not x.flags.contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        x = x.reshape(n_samples, n_dims * self.n_timestep)
        super()._fit(x, y, self.max_samples, self.max_depth, sample_weight)
        return self

    def _make_estimator(self, append=True, random_state=None):
        estimator = super()._make_estimator(append, random_state)
        if self.n_dims > 1:
            estimator.force_dim = self.n_dims
        return estimator


class ShapeletForestClassifier(BaseShapeletForestClassifier):
    def __init__(self,
                 *,
                 n_estimators=100,
                 n_shapelets=10,
                 max_depth=None,
                 min_samples_split=2,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 oob_score=False,
                 bootstrap=True,
                 n_jobs=None,
                 random_state=None):
        super().__init__(
            base_estimator=ShapeletTreeClassifier(),
            estimator_params=(
                "max_depth", "n_shapelets", "min_samples_split", "min_shapelet_size",
                "max_shapelet_size", "metric", "metric_params"
            ),
            n_shapelets=n_shapelets,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            oob_score=oob_score,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state
        )


class ExtraShapeletTreesClassifier(BaseShapeletForestClassifier):
    def __init__(self,
                 *,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 oob_score=False,
                 bootstrap=True,
                 n_jobs=None,
                 random_state=None):
        super().__init__(
            base_estimator=ExtraShapeletTreeClassifier(),
            estimator_params=(
                "max_depth", "min_samples_split", "min_shapelet_size",
                "max_shapelet_size", "metric", "metric_params"
            ),
            n_shapelets=1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            oob_score=oob_score,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state
        )


class BaseShapeletForestRegressor(ShapeletForestMixin, BaggingRegressor):
    def __init__(self,
                 base_estimator,
                 *,
                 estimator_params=tuple(),
                 oob_score=False,
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
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_jobs=n_jobs,
            random_state=random_state,
            oob_score=oob_score
        )
        self.estimator_params = estimator_params
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params

    def _parallel_args(self):
        return _joblib_parallel_args(prefer='threads')

    def predict(self, x, check_input=True):
        x = self._validate_x_predict(x, check_input)
        return super().predict(x)

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

        if x.dtype != np.float64 or not x.flags.contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)

        if y.dtype != np.float64 or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.float64)

        x = x.reshape(n_samples, n_dims * self.n_timestep)
        super()._fit(x, y, self.max_samples, self.max_depth, sample_weight=sample_weight)
        return self

    def _make_estimator(self, append=True, random_state=None):
        estimator = super()._make_estimator(append, random_state)
        if self.n_dims > 1:
            estimator.force_dim = self.n_dims
        return estimator


class ShapeletForestRegressor(BaseShapeletForestRegressor):
    def __init__(self,
                 *,
                 n_estimators=100,
                 n_shapelets=10,
                 max_depth=None,
                 min_samples_split=2,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 oob_score=False,
                 bootstrap=True,
                 n_jobs=None,
                 random_state=None):
        super().__init__(
            base_estimator=ShapeletTreeRegressor(),
            estimator_params=(
                "max_depth", "n_shapelets", "min_samples_split", "min_shapelet_size",
                "max_shapelet_size", "metric", "metric_params"
            ),
            n_shapelets=n_shapelets,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            oob_score=oob_score,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state
        )


class ExtraShapeletTreesRegressor(BaseShapeletForestRegressor):
    def __init__(self,
                 *,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 metric='euclidean',
                 metric_params=None,
                 oob_score=False,
                 bootstrap=True,
                 n_jobs=None,
                 random_state=None):
        super().__init__(
            base_estimator=ExtraShapeletTreeRegressor(),
            estimator_params=(
                "max_depth", "min_samples_split", "min_shapelet_size",
                "max_shapelet_size", "metric", "metric_params"
            ),
            n_shapelets=1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            oob_score=oob_score,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state
        )


class IsolationShapeletForest(OutlierMixin, BaseBagging):
    def __init__(self, *,
                 n_estimators=100,
                 bootstrap=False,
                 n_jobs=None,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 min_samples_split=2,
                 max_samples='auto',
                 contamination='auto',
                 contamination_set="training",
                 warm_start=False,
                 metric='euclidean',
                 metric_params=None,
                 random_state=None):
        super(IsolationShapeletForest, self).__init__(
            base_estimator=ExtraShapeletTreeRegressor(),
            bootstrap=bootstrap,
            bootstrap_features=False,
            warm_start=warm_start,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.estimator_params = (
            "min_samples_split", "min_shapelet_size",
            "max_shapelet_size", "metric", "metric_params"
        )
        self.metric = metric
        self.metric_params = metric_params
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.min_samples_split = min_samples_split
        self.contamination = contamination
        self.contamination_set = contamination_set
        self.max_samples = max_samples
        self.n_dims = None
        self.n_timestep = None

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

    def _set_oob_score(self, x, y):
        raise NotImplementedError("OOB score not supported")

    def _parallel_args(self):
        return _joblib_parallel_args(prefer='threads')

    def fit(self, x, y=None, sample_weight=None, check_input=True):
        random_state = check_random_state(self.random_state)
        if check_input:
            x = check_array(x, dtype=np.float64, allow_nd=True, order="C")

        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = x.shape[0]
        self.n_timestep = x.shape[-1]
        if x.ndim > 2:
            n_dims = x.shape[1]
        else:
            n_dims = 1

        self.n_dims = n_dims

        if x.dtype != np.float64 or not x.flags.contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)

        if n_dims > 1:
            self.base_estimator_.force_dim = n_dims

        x = x.reshape(n_samples, n_dims * self.n_timestep)
        y = random_state.uniform(size=x.shape[0])
        max_depth = int(np.ceil(np.log2(max(x.shape[0], 2))))

        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(x.shape[0], 256)
            else:
                raise ValueError("max_samples (%s) is not supported." % self.max_samples)
        elif isinstance(self.max_samples, numbers.Integral):
            max_samples = min(self.max_samples, x.shape[0])
        else:
            if not 0. < self.max_samples <= 1.0:
                raise ValueError("max_samples must be in (0, 1], got %r" % self.max_samples)
            max_samples = int(self.max_samples * x.shape[0])

        super(IsolationShapeletForest, self)._fit(
            x, y, max_samples=max_samples, max_depth=max_depth, sample_weight=sample_weight)

        if self.contamination == 'auto':
            self.offset_ = -0.5
        elif isinstance(self.contamination, numbers.Real):
            if not 0. < self.contamination <= 1.0:
                raise ValueError("contamination must be in (0, 1], got %r" % self.contamination)
            if self.contamination_set == "training":
                self.offset_ = np.percentile(self.score_samples(x), 100.0 * self.contamination)
            elif self.contamination_set == "oob":
                if not self.bootstrap:
                    raise ValueError("contamination cannot be computed from oob-samples unless bootstrap=True")
                self.offset_ = np.percentile(self._oob_score_samples(x), 100.0 * self.contamination)
            else:
                raise ValueError("contamination_set (%s) is not supported" % self.contamination_set)
        else:
            raise ValueError("max_samples (%s) is not supported." % self.max_samples)

        return self

    def predict(self, x):
        is_inlier = np.ones(x.shape[0])
        is_inlier[self.decision_function(x) < 0] = -1
        return is_inlier

    def decision_function(self, x):
        return self.score_samples(x) - self.offset_

    def score_samples(self, x):
        check_is_fitted(self)
        x = self._validate_x_predict(x, check_input=True)
        return self._score_samples(x, self.estimators_)

    def _oob_score_samples(self, x):
        n_samples = x.shape[0]
        n_bootstrap_samples = n_samples

        score_samples = np.zeros((n_samples,))

        for i in range(x.shape[0]):
            estimators = []
            for estimator, samples in zip(self.estimators_, self.estimators_samples_):
                if i not in samples:
                    estimators.append(estimator)
            score_samples[i] = self._score_samples(x[i].reshape((1, self.n_dims, self.n_timestep)),
                                                   estimators, n_bootstrap_samples)
        return score_samples

    def _score_samples(self, x, estimators, n_samples=None):
        # From: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/ensemble/_iforest.py#L411
        n_samples = n_samples or x.shape[0]
        depths = np.zeros(x.shape[0], order="f")

        for tree in estimators:
            leaves_index = tree.apply(x)
            node_indicator = tree.decision_path(x)
            n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

            depths += np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf) - 1.0
        scores = 2 ** (-depths / (len(estimators) * _average_path_length(np.array([n_samples]))))
        return -scores


def _average_path_length(n_samples_leaf):
    # From: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/ensemble/_iforest.py#L480
    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.
    average_path_length[mask_2] = 1.
    average_path_length[not_mask] = (
            2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
            - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)
