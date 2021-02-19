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
#
# Authors: Isak Samsten
import numbers
from abc import abstractmethod

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import OutlierMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble._bagging import BaseBagging
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted

from scipy import sparse

from ..model_selection.outlier import threshold_score
from ..tree import ExtraShapeletTreeClassifier
from ..tree import ExtraShapeletTreeRegressor
from ..tree import ShapeletTreeClassifier
from ..tree import ShapeletTreeRegressor


class ShapeletForestMixin:
    def apply(self, x):
        x = self._validate_x_predict(x, check_input=True)
        results = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **self._parallel_args(),
        )(delayed(tree.apply)(x, check_input=False) for tree in self.estimators_)

        return np.array(results).T

    def decision_function(self, x):
        x = self._validate_x_predict(x, check_input=True)
        indicators = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer="threads"),
        )(
            delayed(tree.decision_path)(x, check_input=False)
            for tree in self.estimators_
        )

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse.hstack(indicators).tocsr(), n_nodes_ptr

    def _validate_x_predict(self, x, check_input):
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions X.ndim ({})".format(x.ndim))
        if self.n_dims > 1 and x.ndim != 3:
            raise ValueError("illegal input dimensions X.ndim != 3")
        if x.shape[-1] != self.n_timestep:
            raise ValueError(
                "illegal input shape ({} != {})".format(x.shape[-1], self.n_timestep)
            )
        if x.ndim > 2 and x.shape[1] != self.n_dims:
            raise ValueError(
                "illegal input shape ({} != {}".format(x.shape[1], self.n_dims)
            )
        if check_input:
            x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.dtype != np.float64 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)
        x = x.reshape(x.shape[0], self.n_dims * self.n_timestep)
        return x


class BaseShapeletForestClassifier(ShapeletForestMixin, BaggingClassifier):
    """Base class for shapelet forest classifiers.

    Warnings
    --------
    This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(
        self,
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
        metric="euclidean",
        metric_params=None,
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
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
        return _joblib_parallel_args(prefer="threads")

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
        """Fit a random shapelet forest classifier"""
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

        if x.dtype != np.float64 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)

        if not y.flags.c_contiguous:
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
    """An ensemble of random shapelet tree classifiers.

    Examples
    --------

    >>> from wildboar.ensemble import ShapeletForestClassifier
    >>> from wildboar.datasets import load_synthetic_control
    >>> x, y = load_synthetic_control()
    >>> f = ShapeletForestClassifier(n_estimators=100, metric='scaled_euclidean')
    >>> f.fit(x, y)
    >>> y_hat = f.predict(x)
    """

    def __init__(
        self,
        *,
        n_estimators=100,
        n_shapelets=10,
        max_depth=None,
        min_samples_split=2,
        min_shapelet_size=0,
        max_shapelet_size=1,
        metric="euclidean",
        metric_params=None,
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        """Shapelet forest classifier.

        Parameters
        ----------
        n_estimators : int, optional
            The number of estimators

        n_shapelets : int, optional
            The number of shapelets to sample at each node

        bootstrap : bool, optional
            Use bootstrap sampling to fit the base estimators

        n_jobs : int, optional
            The number of processor cores used for fitting the ensemble

        min_shapelet_size : float, optional
            The minimum shapelet size to sample

        max_shapelet_size : float, optional
            The maximum shapelet size to sample

        min_samples_split : int, optional
            The minimum samples required to split the decision trees

        warm_start : bool, optional
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Set the metric used to compute the distance between shapelet and time series

        metric_params : dict, optional
            Parameters passed to the metric construction

        oob_score : bool, optional
            Compute out-of-bag estimates of the ensembles performance.

        random_state : int or RandomState, optional
            Controls the random resampling of the original dataset and the construction of
            the base estimators. Pass an int for reproducible output across multiple
            function calls.
        """
        super().__init__(
            base_estimator=ShapeletTreeClassifier(),
            estimator_params=(
                "max_depth",
                "n_shapelets",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "metric",
                "metric_params",
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
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )


class ExtraShapeletTreesClassifier(BaseShapeletForestClassifier):
    """An ensemble of extremely random shapelet trees for time series regression.

    Examples
    --------

    >>> from wildboar.ensemble import ExtraShapeletTreesClassifier
    >>> from wildboar.datasets import load_synthetic_control
    >>> x, y = load_synthetic_control()
    >>> f = ExtraShapeletTreesClassifier(n_estimators=100, metric='scaled_euclidean')
    >>> f.fit(x, y)
    >>> y_hat = f.predict(x)

    """

    def __init__(
        self,
        *,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_shapelet_size=0,
        max_shapelet_size=1,
        metric="euclidean",
        metric_params=None,
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        """Construct a extra shapelet trees classifier.

        Parameters
        ----------
        n_estimators : int, optional
            The number of estimators

        bootstrap : bool, optional
            Use bootstrap sampling to fit the base estimators

        n_jobs : int, optional
            The number of processor cores used for fitting the ensemble

        min_shapelet_size : float, optional
            The minimum shapelet size to sample

        max_shapelet_size : float, optional
            The maximum shapelet size to sample

        min_samples_split : int, optional
            The minimum samples required to split the decision trees

        warm_start : bool, optional
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Set the metric used to compute the distance between shapelet and time series

        metric_params : dict, optional
            Parameters passed to the metric construction

        random_state : int or RandomState, optional
            Controls the random resampling of the original dataset and the construction of
            the base estimators. Pass an int for reproducible output across multiple
            function calls.
        """
        super().__init__(
            base_estimator=ExtraShapeletTreeClassifier(),
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "metric",
                "metric_params",
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
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )


class BaseShapeletForestRegressor(ShapeletForestMixin, BaggingRegressor):
    """Base class for shapelet forest regressors.

    Warnings
    --------
    This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        *,
        estimator_params=tuple(),
        oob_score=False,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        n_shapelets=10,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
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
        return _joblib_parallel_args(prefer="threads")

    def predict(self, x, check_input=True):
        x = self._validate_x_predict(x, check_input)
        return super().predict(x)

    def fit(self, x, y, sample_weight=None, check_input=True):
        """Fit a random shapelet forest regressor"""
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

        if x.dtype != np.float64 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)

        if y.dtype != np.float64 or not y.flags.c_contiguous:
            y = np.ascontiguousarray(y, dtype=np.float64)

        x = x.reshape(n_samples, n_dims * self.n_timestep)
        super()._fit(
            x, y, self.max_samples, self.max_depth, sample_weight=sample_weight
        )
        return self

    def _make_estimator(self, append=True, random_state=None):
        estimator = super()._make_estimator(append, random_state)
        if self.n_dims > 1:
            estimator.force_dim = self.n_dims
        return estimator


class ShapeletForestRegressor(BaseShapeletForestRegressor):
    """An ensemble of random shapelet regression trees.

    Examples
    --------

    >>> from wildboar.ensemble import ShapeletForestRegressor
    >>> from wildboar.datasets import load_synthetic_control
    >>> x, y = load_synthetic_control()
    >>> f = ShapeletForestRegressor(n_estimators=100, metric='scaled_euclidean')
    >>> f.fit(x, y)
    >>> y_hat = f.predict(x)
    """

    def __init__(
        self,
        *,
        n_estimators=100,
        n_shapelets=10,
        max_depth=None,
        min_samples_split=2,
        min_shapelet_size=0,
        max_shapelet_size=1,
        metric="euclidean",
        metric_params=None,
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        """Shapelet forest regressor.

        Parameters
        ----------
        n_estimators : int, optional
            The number of estimators

        n_shapelets : int, optional
            The number of shapelets to sample at each node

        bootstrap : bool, optional
            Use bootstrap sampling to fit the base estimators

        n_jobs : int, optional
            The number of processor cores used for fitting the ensemble

        min_shapelet_size : float, optional
            The minimum shapelet size to sample

        max_shapelet_size : float, optional
            The maximum shapelet size to sample

        min_samples_split : int, optional
            The minimum samples required to split the decision trees

        warm_start : bool, optional
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Set the metric used to compute the distance between shapelet and time series

        metric_params : dict, optional
            Parameters passed to the metric construction

        oob_score : bool, optional
            Compute out-of-bag estimates of the ensembles performance.

        random_state : int or RandomState, optional
            Controls the random resampling of the original dataset and the construction of
            the base estimators. Pass an int for reproducible output across multiple
            function calls.
        """
        super().__init__(
            base_estimator=ShapeletTreeRegressor(),
            estimator_params=(
                "max_depth",
                "n_shapelets",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "metric",
                "metric_params",
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
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )


class ExtraShapeletTreesRegressor(BaseShapeletForestRegressor):
    """An ensemble of extremely random shapelet trees for time series regression.

    Examples
    --------

    >>> from wildboar.ensemble import ExtraShapeletTreesRegressor
    >>> from wildboar.datasets import load_synthetic_control
    >>> x, y = load_synthetic_control()
    >>> f = ExtraShapeletTreesRegressor(n_estimators=100, metric='scaled_euclidean')
    >>> f.fit(x, y)
    >>> y_hat = f.predict(x)

    """

    def __init__(
        self,
        *,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_shapelet_size=0,
        max_shapelet_size=1,
        metric="euclidean",
        metric_params=None,
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        """Construct a extra shapelet trees regressor.

        Parameters
        ----------
        n_estimators : int, optional
            The number of estimators

        bootstrap : bool, optional
            Use bootstrap sampling to fit the base estimators

        n_jobs : int, optional
            The number of processor cores used for fitting the ensemble

        min_shapelet_size : float, optional
            The minimum shapelet size to sample

        max_shapelet_size : float, optional
            The maximum shapelet size to sample

        min_samples_split : int, optional
            The minimum samples required to split the decision trees

        warm_start : bool, optional
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Set the metric used to compute the distance between shapelet and time series

        metric_params : dict, optional
            Parameters passed to the metric construction

        random_state : int or RandomState, optional
            Controls the random resampling of the original dataset and the construction of
            the base estimators. Pass an int for reproducible output across multiple
            function calls.
        """
        super().__init__(
            base_estimator=ExtraShapeletTreeRegressor(),
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "metric",
                "metric_params",
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
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )


class ShapeletForestEmbedding(BaseShapeletForestRegressor):
    """An ensemble of random shapelet trees

    An unsupervised transformation of a time series dataset
    to a high-dimensional sparse representation. A time series i
    indexed by the leaf that it falls into. This leads to a binary
    coding of a time series with as many ones as trees in the forest.

    The dimensionality of the resulting representation is
    ``<= n_estimators * 2^max_depth``
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        n_shapelets=1,
        max_depth=5,
        min_samples_split=2,
        min_shapelet_size=0,
        max_shapelet_size=1,
        metric="euclidean",
        metric_params=None,
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        sparse_output=True,
        random_state=None,
    ):
        """Construct a shapelet forest embedding.

        Parameters
        ----------

        n_estimators : int, optional
            The number of estimators

        bootstrap : bool, optional
            Use bootstrap sampling to fit the base estimators

        n_jobs : int, optional
            The number of processor cores used for fitting the ensemble

        min_shapelet_size : float, optional
            The minimum shapelet size to sample

        max_shapelet_size : float, optional
            The maximum shapelet size to sample

        min_samples_split : int, optional
            The minimum samples required to split the decision trees

        warm_start : bool, optional
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Set the metric used to compute the distance between shapelet and time series

        metric_params : dict, optional
            Parameters passed to the metric construction

        sparse_output : bool, optional
            Return a sparse CSR-matrix.

        random_state : int or RandomState, optional
            Controls the random resampling of the original dataset and the construction of
            the base estimators. Pass an int for reproducible output across multiple
            function calls.
        """
        super().__init__(
            base_estimator=ExtraShapeletTreeRegressor(),
            estimator_params=(
                "n_shapelets",
                "max_depth",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "metric",
                "metric_params",
            ),
            n_shapelets=n_shapelets,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            oob_score=False,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.sparse_output = sparse_output

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported")

    def fit(self, x, y=None, sample_weight=None, check_input=True):
        self.fit_transform(x, y, sample_weight, check_input)
        return self

    def fit_transform(self, x, y=None, sample_weight=None, check_input=True):
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

        if x.dtype != np.float64 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)

        if n_dims > 1:
            self.base_estimator_.force_dim = n_dims

        x = x.reshape(n_samples, n_dims * self.n_timestep)
        y = random_state.uniform(size=x.shape[0])
        super().fit(x, y, sample_weight=sample_weight)
        self.one_hot_encoder_ = OneHotEncoder(
            sparse=self.sparse_output, handle_unknown="ignore"
        )
        return self.one_hot_encoder_.fit_transform(self.apply(x))

    def transform(self, x):
        check_is_fitted(self)
        return self.one_hot_encoder_.transform(self.apply(x))


class IsolationShapeletForest(ShapeletForestMixin, OutlierMixin, BaseBagging):
    """A isolation shapelet forest.

    .. versionadded:: 0.3.5

    Attributes
    ----------
    offset_ : float
        The offset for computing the final decision

    Examples
    --------

    >>> from wildboar.ensemble import IsolationShapeletForest
    >>> from wildboar.datasets import load_two_lead_ecg
    >>> from model_selection.outlier import train_test_split
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> x, y = load_two_lead_ecg("two_lead_ecg")
    >>> x_train, x_test, y_train, y_test = train_test_split(x, y, 1, test_size=0.2, anomalies_train_size=0.05)
    >>> f = IsolationShapeletForest(n_estimators=100, contamination=balanced_accuracy_score)
    >>> f.fit(x_train, y_train)
    >>> y_pred = f.predict(x_test)
    >>> balanced_accuracy_score(y_test, y_pred)

    Or using default offset threshold

    >>> from wildboar.ensemble import IsolationShapeletForest
    >>> from wildboar.datasets import load_two_lead_ecg
    >>> from model_selection.outlier import train_test_split
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> f = IsolationShapeletForest()
    >>> x, y = load_two_lead_ecg("two_lead_ecg")
    >>> x_train, x_test, y_train, y_test = train_test_split(x, y, 1, test_size=0.2, anomalies_train_size=0.05)
    >>> f.fit(x_train)
    >>> y_pred = f.predict(x_test)
    >>> balanced_accuracy_score(y_test, y_pred)
    """

    def __init__(
        self,
        *,
        n_estimators=100,
        bootstrap=False,
        n_jobs=None,
        min_shapelet_size=0,
        max_shapelet_size=1,
        min_samples_split=2,
        max_samples="auto",
        contamination="auto",
        contamination_set="training",
        warm_start=False,
        metric="euclidean",
        metric_params=None,
        random_state=None,
    ):
        """Construct a shapelet isolation forest

        Parameters
        ----------
        n_estimators : int, optional
            The number of estimators

        bootstrap : bool, optional
            Use bootstrap sampling to fit the base estimators

        n_jobs : int, optional
            The number of processor cores used for fitting the ensemble

        min_shapelet_size : float, optional
            The minimum shapelet size to sample

        max_shapelet_size : float, optional
            The maximum shapelet size to sample

        min_samples_split : int, optional
            The minimum samples required to split the decision trees

        max_samples : float or int
            The number of samples to draw to train each base estimator

        contamination : str, float or callable
            The strategy for computing the offset (see `offset_`)

            - if 'auto' ``offset_=-0.5``
            - if 'auc' ``offset_`` is computed as the offset that maximizes the
              area under ROC in the training or out-of-bag set (see ``contamination_set``).
            - if 'prc' ``offset_`` is computed as the offset that maximizes the
              area under PRC in the training or out-of-bag set (see ``contamination_set``)
            - if callable ``offset_`` is computed as the offset that maximizes the score
              computed by the callable in training or out-of-bag set (see ``contamination_set``)
            - if float ``offset_`` is computed as the c:th percentile of scores in the training
              or out-of-bag set (see ``contamination_set``)

            Setting contamination to either 'auc' or 'prc' require that `y` is passed to `fit`.

        contamination_set : {'training', 'oob'}, optional
            Compute the ``offset_`` from either the out-of-bag samples or the training samples.
            'oob' require `bootstrap=True`.

        warm_start : bool, optional
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Set the metric used to compute the distance between shapelet and time series

        metric_params : dict, optional
            Parameters passed to the metric construction

        random_state : int or RandomState, optional
            Controls the random resampling of the original dataset and the construction of
            the base estimators. Pass an int for reproducible output across multiple
            function calls.
        """
        super(IsolationShapeletForest, self).__init__(
            base_estimator=ExtraShapeletTreeRegressor(),
            bootstrap=bootstrap,
            bootstrap_features=False,
            warm_start=warm_start,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.estimator_params = (
            "min_samples_split",
            "min_shapelet_size",
            "max_shapelet_size",
            "metric",
            "metric_params",
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

    def _set_oob_score(self, x, y):
        raise NotImplementedError("OOB score not supported")

    def _parallel_args(self):
        return _joblib_parallel_args(prefer="threads")

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

        if x.dtype != np.float64 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)

        if n_dims > 1:
            self.base_estimator_.force_dim = n_dims

        x = x.reshape(n_samples, n_dims * self.n_timestep)
        rnd_y = random_state.uniform(size=x.shape[0])
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(x.shape[0], 256)
            else:
                raise ValueError(
                    "max_samples (%s) is not supported." % self.max_samples
                )
        elif isinstance(self.max_samples, numbers.Integral):
            max_samples = min(self.max_samples, x.shape[0])
        else:
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError(
                    "max_samples must be in (0, 1], got %r" % self.max_samples
                )
            max_samples = int(self.max_samples * x.shape[0])

        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super(IsolationShapeletForest, self)._fit(
            x,
            rnd_y,
            max_samples=max_samples,
            max_depth=max_depth,
            sample_weight=sample_weight,
        )

        self.max_samples_ = max_samples
        if self.contamination == "auto":
            self.offset_ = -0.5
        elif self.contamination in ["auc", "prc"] or hasattr(
            self.contamination, "__call__"
        ):
            if y is None:
                raise ValueError(
                    "contamination cannot be computed without training labels"
                )

            if self.contamination_set == "oob":
                if not self.bootstrap:
                    raise ValueError(
                        "contamination cannot be computed from oob-samples unless bootstrap=True"
                    )
                scores = self._oob_score_samples(x)
            else:
                scores = self.score_samples(x)

            if self.contamination == "auc":
                fpr, tpr, thresholds = roc_curve(y, scores)
                best_threshold = np.argmax(tpr - fpr)
            elif self.contamination == "prc":
                precision, recall, thresholds = precision_recall_curve(y, scores)
                fscore = (2 * precision * recall) / (precision + recall)
                best_threshold = np.argmax(fscore)
            else:
                score = threshold_score(y, scores, self.contamination)
                best_threshold = np.argmax(score)
                thresholds = scores
            self.offset_ = thresholds[best_threshold]
        elif isinstance(self.contamination, numbers.Real):
            if not 0.0 < self.contamination <= 1.0:
                raise ValueError(
                    "contamination must be in (0, 1], got %r" % self.contamination
                )
            if self.contamination_set == "training":
                self.offset_ = np.percentile(
                    self.score_samples(x), 100.0 * self.contamination
                )
            elif self.contamination_set == "oob":
                if not self.bootstrap:
                    raise ValueError(
                        "contamination cannot be computed from oob-samples unless bootstrap=True"
                    )
                self.offset_ = np.percentile(
                    self._oob_score_samples(x), 100.0 * self.contamination
                )
            else:
                raise ValueError(
                    "contamination_set (%s) is not supported" % self.contamination_set
                )
        else:
            raise ValueError(
                "contamination (%s) is not supported." % self.contamination
            )

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
        return _score_samples(x, self.estimators_, self.max_samples_)

    def _oob_score_samples(self, x):
        n_samples = x.shape[0]
        score_samples = np.zeros((n_samples,))

        for i in range(x.shape[0]):
            estimators = []
            for estimator, samples in zip(self.estimators_, self.estimators_samples_):
                if i not in samples:
                    estimators.append(estimator)
            score_samples[i] = _score_samples(
                x[i].reshape((1, self.n_dims, self.n_timestep)),
                estimators,
                self.max_samples_,
            )
        return score_samples


def _score_samples(x, estimators, max_samples):
    # From: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/ensemble/_iforest.py#L411
    depths = np.zeros(x.shape[0], order="f")

    for tree in estimators:
        leaves_index = tree.apply(x)
        node_indicator = tree.decision_path(x)
        n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

        depths += (
            np.ravel(node_indicator.sum(axis=1))
            + _average_path_length(n_samples_leaf)
            - 1.0
        )
    scores = 2 ** (
        -depths / (len(estimators) * _average_path_length(np.array([max_samples])))
    )
    return -scores


def _average_path_length(n_samples_leaf):
    # From: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/ensemble/_iforest.py#L480
    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)
