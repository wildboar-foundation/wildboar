# Authors: Isak Samsten
# License: BSD 3 clause

import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, _fit_context
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.extmath import softmax
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, check_random_state

from ..base import BaseEstimator
from ..datasets.preprocess import SparseScaler


def _transform_has_attr(attr):
    def check(self):
        return hasattr(self.pipe_["transform"], "embedding_")

    return check


class BaseTransformEstimator(BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints = {
        "random_state": ["random_state"],
        "n_jobs": [None, numbers.Integral],
    }

    def __init__(self, *, random_state=None, n_jobs=None):
        self.random_state = random_state
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, x, y, sample_weight=None):
        x, y = self._validate_data(x, y, dtype=float, allow_3d=True)
        self.pipe_ = Pipeline(self._build_pipeline())

        # TODO: apply sample weights to the transform
        self.pipe_.fit(x, y, estimator__sample_weight=sample_weight)
        return self

    def _build_pipeline(self):
        random_state = check_random_state(self.random_state)
        return [
            ("transform", self._get_transform(random_state.randint(2**31))),
            ("estimator", self._get_estimator(random_state.randint(2**31))),
        ]

    @abstractmethod
    def _get_transform(self, random_state):
        pass

    @abstractmethod
    def _get_estimator(self, random_state):
        pass


def _pipe_has(attr):
    def check(self):
        hasattr(self.pipe_, attr)

    return check


class BaseTransformClassifier(ClassifierMixin, BaseTransformEstimator):
    def predict(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, allow_3d=True)
        return self.pipe_.predict(x)

    @available_if(_pipe_has("predict_proba"))
    def predict_proba(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, allow_3d=True)
        return self.pipe_.predict_proba(x)

    @available_if(_pipe_has("predict_log_proba"))
    def predict_log_proba(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, allow_3d=True)
        return self.pipe_.predict_log_proba(x)

    def decision_function(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, allow_3d=True)
        return self.pipe_.decision_function(x)

    @property
    def classes_(self):
        return self.pipe_.classes_


class BaseTransformRegressor(RegressorMixin, BaseTransformEstimator):
    def predict(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, allow_3d=True)
        return self.pipe_.predict(x)

    @available_if(_pipe_has("decision_function"))
    def decision_function(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, allow_3d=True)
        return self.pipe_.decision_function(x)


class TransformRidgeClassifierCV(BaseTransformClassifier):
    _parameter_constraints = {
        **BaseTransformEstimator._parameter_constraints,
        **RidgeClassifierCV._parameter_constraints,
        "normalize": [bool, StrOptions({"sparse"})],
    }
    _parameter_constraints.pop("store_cv_values")
    _parameter_constraints.pop("store_cv_results")

    def __init__(
        self,
        *,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        scoring=None,
        cv=None,
        class_weight=None,
        normalize=True,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.cv = cv
        self.normalize = normalize
        self.class_weight = class_weight
        self.random_state = random_state

    def _get_estimator(self, random_state):
        return RidgeClassifierCV(
            alphas=self.alphas,
            fit_intercept=self.fit_intercept,
            scoring=self.scoring,
            cv=self.cv,
            class_weight=self.class_weight,
            store_cv_values=False,
        )

    def _build_pipeline(self):
        pipeline = super()._build_pipeline()
        if self.normalize is True:
            pipeline.insert(1, ("normalize", StandardScaler()))
        elif self.normalize == "sparse":
            pipeline.insert(1, ("normalize", SparseScaler()))

        return pipeline

    def predict_proba(self, x):
        decision = self.decision_function(x)
        if decision.ndim == 1:
            decision_2d = np.c_[-decision, decision]
        else:
            decision_2d = decision
        return softmax(decision_2d, copy=False)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
                "_check_sample_weights_invariance_samples_order": ("test"),
            }
        }


class TransformRidgeCV(BaseTransformRegressor):
    _parameter_constraints = {
        **BaseTransformEstimator._parameter_constraints,
        **RidgeCV._parameter_constraints,
        "normalize": [bool, StrOptions({"sparse"})],
    }

    for param in ("store_cv_values", "store_cv_results", "alpha_per_target"):
        _parameter_constraints.pop(param)

    def __init__(
        self,
        *,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        scoring=None,
        cv=None,
        gcv_mode=None,
        normalize=True,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.cv = cv
        self.normalize = normalize
        self.gcv_mode = gcv_mode

    def _get_estimator(self, random_state):
        return RidgeCV(
            alphas=self.alphas,
            fit_intercept=self.fit_intercept,
            scoring=self.scoring,
            cv=self.cv,
            gcv_mode=self.gcv_mode,
            store_cv_values=False,
            alpha_per_target=False,
        )

    def _build_pipeline(self):
        pipeline = super()._build_pipeline()
        if self.normalize is True:
            pipeline.insert(1, ("normalize", StandardScaler()))
        elif self.normalize == "sparse":
            pipeline.insert(1, ("normalize", SparseScaler()))

        return pipeline

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
                "_check_sample_weights_invariance_samples_order": ("test"),
            }
        }
