# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import OutlierMixin
from sklearn.ensemble import BaggingClassifier as SklearnBaggingClassifier
from sklearn.ensemble import BaggingRegressor as SklearnBaggingRegressor
from sklearn.ensemble._bagging import BaseBagging as SklearnBaseBagging
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.validation import check_is_fitted, check_scalar

from ..base import BaseEstimator
from ..tree import (
    ExtraShapeletTreeClassifier,
    ExtraShapeletTreeRegressor,
    IntervalTreeClassifier,
    IntervalTreeRegressor,
    PivotTreeClassifier,
    ProximityTreeClassifier,
    ShapeletTreeClassifier,
    ShapeletTreeRegressor,
)
from ..tree._tree import RocketTreeClassifier, RocketTreeRegressor


class BaseBagging(BaseEstimator, SklearnBaseBagging, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        bootstrap=True,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=bootstrap,
            bootstrap_features=False,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _validate_data(
        self,
        x="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        **check_params,
    ):
        new_check_params = {"allow_3d": True}
        if "y_numeric" in check_params:
            new_check_params["y_numeric"] = check_params["y_numeric"]

        out = super()._validate_data(
            x,
            y,
            reset=reset,
            validate_separately=validate_separately,
            dtype=float,
            **new_check_params,
        )
        no_val_X = isinstance(x, str) and x == "no_validation"
        no_val_y = y is None or isinstance(y, str) and y == "no_validation"
        if no_val_X:
            return out
        elif no_val_y:
            x = out
        else:
            x, y = out

        if self.n_dims_in_ > 1:
            x = x.reshape(x.shape[0], -1)

        if no_val_y:
            return x
        else:
            return x, y

    def fit(self, x, y, sample_weight=None):
        x, y = self._validate_data(x, y, allow_3d=True, dtype=float)
        self._fit(
            x, y, self.max_samples, self.max_depth, sample_weight, check_input=False
        )
        return self

    def _make_estimator(self, append=True, random_state=None):
        estimator = super()._make_estimator(append, random_state)
        if self.n_dims_in_ > 1:
            estimator._force_n_dims = self.n_dims_in_
        return estimator

    def _more_tags(self):
        return {"X_types": ["2darray", "3darray"]}


class ForestMixin:
    def apply(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, allow_3d=True, reset=False)
        results = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **{"prefer": "threads"},
        )(delayed(tree.apply)(x, check_input=False) for tree in self.estimators_)

        return np.array(results).T

    def decision_path(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, allow_3d=True, reset=False)
        indicators = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **{"prefer": "threads"},
        )(
            delayed(tree.decision_path)(x, check_input=False)
            for tree in self.estimators_
        )

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse.hstack(indicators).tocsr(), n_nodes_ptr


class BaggingClassifier(BaseBagging, SklearnBaggingClassifier):
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        bootstrap=True,
        oob_score=False,
        class_weight=None,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):

        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.class_weight = class_weight

    def fit(self, x, y, sample_weight=None):
        x, y = self._validate_data(x, y, allow_3d=True, dtype=float)

        if self.class_weight is not None:
            class_weight = compute_sample_weight(self.class_weight, y)
            if sample_weight is not None:
                sample_weight = sample_weight * class_weight
            else:
                sample_weight = class_weight

        self._fit(
            x, y, self.max_samples, self.max_depth, sample_weight, check_input=True
        )
        return self


class BaseForestClassifier(ForestMixin, BaggingClassifier, metaclass=ABCMeta):
    """"""

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        estimator_params=tuple(),
        *,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.estimator_params = estimator_params

    def _fit(
        self,
        X,
        y,
        max_samples=None,
        max_depth=None,
        sample_weight=None,
        check_input=False,
    ):
        return super()._fit(X, y, max_samples, max_depth, sample_weight, False)

    def _parallel_args(self):
        return {"prefer": "threads"}


class BaseShapeletForestClassifier(BaseForestClassifier, metaclass=ABCMeta):
    """Base class for shapelet forest classifiers.

    Warnings
    --------
    This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        estimator_params=tuple(),
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        n_shapelets="warn",
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=base_estimator,
            estimator_params=estimator_params,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            class_weight=class_weight,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
        )
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params


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
        n_estimators=100,
        *,
        n_shapelets="warn",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        alpha=None,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        class_weight=None,
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

        alpha : float, optional
            Dynamically decrease the number of sampled shapelets at each node according
            to the current depth.

            .. math:`w = 1 - e^{-|alpha| * depth})`

            - if :math:`alpha < 0`, the number of sampled shapelets decrease from
              ``n_shapelets`` towards 1 with increased depth.

              .. math:`n_shapelets * (1 - w)`

            - if :math:`alpha > 0`, the number of sampled shapelets increase from ``1``
              towards ``n_shapelets`` with increased depth.

              .. math:`n_shapelets * w`

            - if ``None``, the number of sampled shapelets are the same independeth of
              depth.

        min_samples_split : int, optional
            The minimum samples required to split the decision trees

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf

        criterion : {"entropy", "gini"}, optional
            The criterion used to evaluate the utility of a split

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value

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

        class_weight : dict or "balanced", optional
            Weights associated with the labels

            - if dict, weights on the form {label: weight}
            - if "balanced" each class weight inversely proportional to the class
              frequency
            - if None, each class has equal weight

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(
            base_estimator=ShapeletTreeClassifier(),
            estimator_params=(
                "max_depth",
                "n_shapelets",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "min_shapelet_size",
                "max_shapelet_size",
                "alpha",
                "metric",
                "metric_params",
                "criterion",
            ),
            n_shapelets=n_shapelets,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            oob_score=oob_score,
            bootstrap=bootstrap,
            warm_start=warm_start,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.alpha = alpha


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
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        class_weight=None,
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

        min_samples_leaf : int, optional
            The minimum number of samples in a leaf

        criterion : {"entropy", "gini"}, optional
            The criterion used to evaluate the utility of a split

        min_impurity_decrease : float, optional
            A split will be introduced only if the impurity decrease is larger than or
            equal to this value

        warm_start : bool, optional
            When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Set the metric used to compute the distance between shapelet and time series

        metric_params : dict, optional
            Parameters passed to the metric construction

        class_weight : dict or "balanced", optional
            Weights associated with the labels

            - if dict, weights on the form {label: weight}
            - if "balanced" each class weight inversely proportional to the class
              frequency
            - if None, each class has equal weight

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(
            base_estimator=ExtraShapeletTreeClassifier(),
            estimator_params=(
                "max_depth",
                "min_impurity_decrease",
                "min_samples_leaf",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "metric",
                "metric_params",
                "criterion",
            ),
            n_shapelets=1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            oob_score=oob_score,
            bootstrap=bootstrap,
            warm_start=warm_start,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )


class BaggingRegressor(BaseBagging, SklearnBaggingRegressor):
    def __init__(
        self,
        base_estimator=None,
        n_estimators=100,
        *,
        max_samples=1.0,
        bootstrap=True,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            base_estimator,
            n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, x, y, sample_weight=None):
        x, y = self._validate_data(x, y, allow_3d=True, dtype=float, y_numeric=True)

        super()._fit(x, y, self.max_samples, self.max_depth, sample_weight)
        return self


class BaseForestRegressor(ForestMixin, BaggingRegressor, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        *,
        base_estimator,
        estimator_params=tuple(),
        oob_score=False,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        criterion="squared_error",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
        )
        self.estimator_params = estimator_params
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _fit(
        self,
        X,
        y,
        max_samples=None,
        max_depth=None,
        sample_weight=None,
        check_input=False,
    ):
        return super()._fit(X, y, max_samples, max_depth, sample_weight, False)

    def _parallel_args(self):
        return {"prefer": "threads"}


class BaseShapeletForestRegressor(BaseForestRegressor):
    """Base class for shapelet forest classifiers.

    Warnings
    --------
    This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(
        self,
        *,
        base_estimator,
        estimator_params=tuple(),
        oob_score=False,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        n_shapelets="warn",
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=base_estimator,
            estimator_params=estimator_params,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
        )
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params

    def _parallel_args(self):
        return {"prefer": "threads"}


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
        n_estimators=100,
        *,
        n_shapelets="warn",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        alpha=None,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
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

        alpha : float, optional
            Dynamically decrease the number of sampled shapelets at each node according
            to the current depth.

            .. math:`w = 1 - e^{-|alpha| * depth})`

            - if :math:`alpha < 0`, the number of sampled shapelets decrease from
              ``n_shapelets`` towards 1 with increased depth.

              .. math:`n_shapelets * (1 - w)`

            - if :math:`alpha > 0`, the number of sampled shapelets increase from ``1``
              towards ``n_shapelets`` with increased depth.

              .. math:`n_shapelets * w`

            - if ``None``, the number of sampled shapelets are the same independeth of
              depth.

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

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(
            base_estimator=ShapeletTreeRegressor(),
            estimator_params=(
                "max_depth",
                "n_shapelets",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "min_shapelet_size",
                "max_shapelet_size",
                "alpha",
                "metric",
                "metric_params",
                "criterion",
            ),
            n_shapelets=n_shapelets,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            oob_score=oob_score,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.alpha = alpha


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
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_shapelet_size=0,
        max_shapelet_size=1,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
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

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(
            base_estimator=ExtraShapeletTreeRegressor(),
            estimator_params=(
                "max_depth",
                "min_impurity_decrease",
                "min_samples_leaf",
                "min_samples_split",
                "min_shapelet_size",
                "max_shapelet_size",
                "metric",
                "metric_params",
                "criterion",
            ),
            n_shapelets=1,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
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
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        sparse_output=True,
        random_state=None,
    ):
        """
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

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(
            base_estimator=ExtraShapeletTreeRegressor(),
            estimator_params=(
                "n_shapelets",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "min_shapelet_size",
                "max_shapelet_size",
                "metric",
                "metric_params",
                "criterion",
            ),
            n_shapelets=n_shapelets,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            metric=metric,
            metric_params=metric_params,
            criterion=criterion,
            oob_score=False,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.sparse_output = sparse_output

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported")

    def fit(self, x, y=None, sample_weight=None):
        self.fit_transform(x, y, sample_weight)
        return self

    def fit_transform(self, x, y=None, sample_weight=None):
        x = self._validate_data(x, allow_3d=True, dtype=float)
        random_state = check_random_state(self.random_state)
        y = random_state.uniform(size=x.shape[0])
        super().fit(x, y, sample_weight=sample_weight)
        self.one_hot_encoder_ = OneHotEncoder(
            sparse=self.sparse_output, handle_unknown="ignore"
        )
        return self.one_hot_encoder_.fit_transform(self.apply(x))

    def transform(self, x):
        check_is_fitted(self)
        return self.one_hot_encoder_.transform(self.apply(x))


class IsolationShapeletForest(OutlierMixin, ForestMixin, BaseBagging):
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
    >>> from wildboar.model_selection.outlier import train_test_split
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> x, y = load_two_lead_ecg()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...    x, y, 1, test_size=0.2, anomalies_train_size=0.05
    ... )
    >>> f = IsolationShapeletForest(
    ...     n_estimators=100, contamination=balanced_accuracy_score
    ... )
    >>> f.fit(x_train, y_train)
    >>> y_pred = f.predict(x_test)
    >>> balanced_accuracy_score(y_test, y_pred)

    Or using default offset threshold

    >>> from wildboar.ensemble import IsolationShapeletForest
    >>> from wildboar.datasets import load_two_lead_ecg
    >>> from wildboar.model_selection.outlier import train_test_split
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> f = IsolationShapeletForest()
    >>> x, y = load_two_lead_ecg()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     x, y, 1, test_size=0.2, anomalies_train_size=0.05
    ... )
    >>> f.fit(x_train)
    >>> y_pred = f.predict(x_test)
    >>> balanced_accuracy_score(y_test, y_pred)

    """

    def __init__(
        self,
        *,
        n_shapelets=1,
        n_estimators=100,
        bootstrap=False,
        n_jobs=None,
        min_shapelet_size=0,
        max_shapelet_size=1,
        min_samples_split=2,
        max_samples="auto",
        contamination="auto",
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

        max_samples : "auto", float or int, optional
            The number of samples to draw to train each base estimator

        contamination : 'auto' or float, optional
            The strategy for computing the offset (see `offset_`)

            - if 'auto', `offset_=-0.5`

            - if float ``offset_`` is computed as the c:th percentile of scores.

            If `bootstrap=True`, out-of-bag samples are used for computing the scores.

        warm_start : bool, optional
            When set to True, reuse the solution of the previous call to fit and add
            more estimators to the ensemble, otherwise, just fit a whole new ensemble.

        metric : {'euclidean', 'scaled_euclidean', 'scaled_dtw'}, optional
            Set the metric used to compute the distance between shapelet and time series

        metric_params : dict, optional
            Parameters passed to the metric construction

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super(IsolationShapeletForest, self).__init__(
            base_estimator=ExtraShapeletTreeRegressor(),
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.estimator_params = (
            "n_shapelets",
            "min_samples_split",
            "min_shapelet_size",
            "max_shapelet_size",
            "metric",
            "metric_params",
        )
        self.n_shapelets = n_shapelets
        self.metric = metric
        self.metric_params = metric_params
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.min_samples_split = min_samples_split
        self.contamination = contamination
        self.max_samples = max_samples

    def _set_oob_score(self, x, y):
        raise NotImplementedError("OOB score not supported")

    def _parallel_args(self):
        return {"prefer": "threads"}

    def fit(self, x, y=None, sample_weight=None):
        x = self._validate_data(x, allow_3d=True, dtype=float)
        random_state = check_random_state(self.random_state)

        rnd_y = random_state.uniform(size=x.shape[0])
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(x.shape[0], 256)
            else:
                raise ValueError(
                    "max_samples must be 'auto', int or float, got %r."
                    % self.max_samples
                )
        elif isinstance(self.max_samples, numbers.Integral):
            max_samples = min(self.max_samples, x.shape[0])
        else:
            max_samples = math.ceil(
                check_scalar(
                    self.max_samples,
                    "max_samples",
                    numbers.Real,
                    min_val=0,
                    max_val=1.0,
                    include_boundaries="right",
                )
                * x.shape[0]
            )

        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super(IsolationShapeletForest, self)._fit(
            x,
            rnd_y,
            max_samples=max_samples,
            max_depth=max_depth,
            sample_weight=sample_weight,
            check_input=False,
        )

        self.max_samples_ = max_samples
        if self.contamination == "auto":
            self.offset_ = -0.5
        elif isinstance(self.contamination, numbers.Real):
            if not 0 < self.contamination <= 0.5:
                raise ValueError("contamination must be in (0, 0.5]")
            if self.bootstrap:
                scores = self._oob_score_samples(x)
            else:
                scores = self.score_samples(x)

            self.offset_ = np.percentile(scores, 100.0 * self.contamination)
        else:
            raise ValueError(
                "contamination must be 'auto' or float, got %r." % self.contamination
            )

        return self

    def predict(self, x):
        decision = self.decision_function(x)
        is_inlier = np.ones(decision.shape[0], dtype=int)
        is_inlier[decision < 0] = -1
        return is_inlier

    def decision_function(self, x):
        return self.score_samples(x) - self.offset_

    def score_samples(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, reset=False, allow_3d=True, dtype=float)
        return _score_samples(x, self.estimators_, self.max_samples_)

    def _oob_score_samples(self, x):
        n_samples = x.shape[0]
        score_samples = np.zeros((n_samples,))

        # We need to make this more efficient
        for i in range(x.shape[0]):
            estimators = []
            for estimator, samples in zip(self.estimators_, self.estimators_samples_):
                if i not in samples:
                    estimators.append(estimator)
            score_samples[i] = _score_samples(x[[i]], estimators, self.max_samples_)
        return score_samples

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }


def _score_samples(x, estimators, max_samples):
    # From: https://github.com/scikit-learn/scikit-learn/blob/
    # 0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/ensemble/_iforest.py#L411
    depths = np.zeros(x.shape[0], order="F")

    for tree in estimators:
        leaves_index = tree.apply(x, check_input=False)
        node_indicator = tree.decision_path(x, check_input=False)
        n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

        depths += (
            np.ravel(node_indicator.sum(axis=1))
            + _average_path_length(n_samples_leaf)
            - 1.0
        )

    denominator = len(estimators) * _average_path_length(np.array([max_samples]))
    scores = 2 ** -np.divide(
        depths, denominator, out=np.ones_like(depths), where=denominator != 0
    )
    return -scores


def _average_path_length(n_samples_leaf):
    # From: https://github.com/scikit-learn/scikit-learn/blob/
    # 0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/ensemble/_iforest.py#L480
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


class RockestRegressor(BaseForestRegressor):
    """An ensemble of rocket tree regressors."""

    def __init__(
        self,
        n_estimators=100,
        *,
        n_kernels=10,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        sampling="auto",
        sampling_params=None,
        kernel_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        criterion="squared_error",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=RocketTreeRegressor(),
            estimator_params=(
                "n_kernels",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "kernel_size",
                "sampling",
                "sampling_params",
                "bias_prob",
                "normalize_prob",
                "padding_prob",
                "criterion",
            ),
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
        )
        self.n_kernels = n_kernels
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.bias_prob = bias_prob
        self.normalize_prob = normalize_prob
        self.padding_prob = padding_prob

    def _parallel_args(self):
        return {"prefer": "threads"}


class RockestClassifier(BaseForestClassifier):
    """An ensemble of rocket tree classifiers."""

    def __init__(
        self,
        n_estimators=100,
        *,
        n_kernels=10,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        sampling="auto",
        sampling_params=None,
        kernel_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        class_weight=None,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=RocketTreeClassifier(),
            estimator_params=(
                "n_kernels",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "kernel_size",
                "sampling",
                "sampling_params",
                "bias_prob",
                "normalize_prob",
                "padding_prob",
                "criterion",
            ),
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state,
            oob_score=oob_score,
            class_weight=class_weight,
        )
        self.n_kernels = n_kernels
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.bias_prob = bias_prob
        self.normalize_prob = normalize_prob
        self.padding_prob = padding_prob

    def _parallel_args(self):
        return {"prefer": "threads"}


class IntervalForestClassifier(BaseForestClassifier):
    """An ensemble of interval tree classifiers."""

    def __init__(
        self,
        n_estimators=100,
        *,
        n_intervals="sqrt",
        intervals="fixed",
        summarizer="auto",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=IntervalTreeClassifier(),
            estimator_params=(
                "n_intervals",
                "intervals",
                "summarizer",
                "sample_size",
                "min_size",
                "max_size",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "criterion",
            ),
            oob_score=oob_score,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.n_intervals = n_intervals
        self.intervals = intervals
        self.summarizer = summarizer
        self.sample_size = sample_size
        self.min_size = min_size
        self.max_size = max_size

    def _parallel_args(self):
        return {"prefer": "threads"}


class IntervalForestRegressor(BaseForestRegressor):
    """An ensemble of interval tree regressors."""

    def __init__(
        self,
        n_estimators=100,
        *,
        n_intervals="sqrt",
        intervals="fixed",
        summarizer="auto",
        sample_size=0.5,
        min_size=0.0,
        max_size=1.0,
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        criterion="squared_error",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=IntervalTreeRegressor(),
            estimator_params=(
                "n_intervals",
                "intervals",
                "summarizer",
                "sample_size",
                "min_size",
                "max_size",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "criterion",
            ),
            oob_score=oob_score,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.n_intervals = n_intervals
        self.intervals = intervals
        self.summarizer = summarizer
        self.sample_size = sample_size
        self.min_size = min_size
        self.max_size = max_size

    def _parallel_args(self):
        return {"prefer": "threads"}


class PivotForestClassifier(BaseForestClassifier):
    """An ensemble of interval tree classifiers."""

    def __init__(
        self,
        n_estimators=100,
        *,
        n_pivot="sqrt",
        metrics="all",
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=PivotTreeClassifier(),
            estimator_params=(
                "n_pivot",
                "metrics",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "criterion",
            ),
            oob_score=oob_score,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.n_pivot = n_pivot
        self.metrics = metrics

    def _parallel_args(self):
        return {"prefer": "threads"}


class ProximityForestClassifier(BaseForestClassifier):
    """A forest of proximity trees

    References
    ----------
    Lucas, Benjamin, Ahmed Shifaz, Charlotte Pelletier, Lachlan O'Neill, Nayyar Zaidi, \
    Bart Goethals, François Petitjean, and Geoffrey I. Webb. (2019)
        Proximity forest: an effective and scalable distance-based classifier for time
        series. Data Mining and Knowledge Discovery
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        n_pivot=1,
        pivot_sample="label",
        metric_sample="weighted",
        metric_factories="default",
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(
            base_estimator=ProximityTreeClassifier(),
            estimator_params=(
                "n_pivot",
                "metric_factories",
                "pivot_sample",
                "metric_sample",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "criterion",
            ),
            oob_score=oob_score,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            n_jobs=n_jobs,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.n_pivot = n_pivot
        self.metric_factories = metric_factories
        self.pivot_sample = pivot_sample
        self.metric_sample = metric_sample

    def _parallel_args(self):
        return {"prefer": "threads"}
