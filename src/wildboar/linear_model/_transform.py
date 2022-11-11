# Authors: Isak Samsten
# License: BSD 3 clause

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted, check_random_state

from ..base import BaseEstimator


class BaseTransformEstimator(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, *, random_state=None, n_jobs=None):
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, x, y, sample_weight=None):
        x, y = self._validate_data(x, y, dtype=float, allow_3d=True)
        random_state = check_random_state(self.random_state)
        self.pipe_ = Pipeline(
            [
                ("transform", self._get_transform(random_state.randint(2**31))),
                ("estimator", self._get_estimator(random_state.randint(2**31))),
            ],
        )
        self.pipe_.fit(x, y, estimator__sample_weight=sample_weight)
        return self

    @abstractmethod
    def _get_transform(self, random_state):
        pass

    @abstractmethod
    def _get_estimator(self, random_state):
        pass


class TransformClassifierMixin:
    def predict(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, allow_3d=True)
        return self.pipe_.predict(x)

    def predict_proba(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, allow_3d=True)
        return self.pipe_.predict_proba(x)

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
        return self.pipe_[-1].classes_


class TransformRegressorMixin:
    def predict(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, allow_3d=True)
        return self.pipe_.predict(x)

    def decision_function(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, dtype=float, allow_3d=True)
        return self.pipe_.decision_function(x)


class TransformRidgeClassifierCV(
    ClassifierMixin, TransformClassifierMixin, BaseTransformEstimator
):
    def __init__(
        self,
        *,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        normalize="deprecated",
        scoring=None,
        cv=None,
        class_weight=None,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.cv = cv
        self.class_weight = class_weight
        self.random_state = random_state

    def _get_estimator(self, random_state):
        if self.normalize != "deprecated":
            warnings.warn(
                "normalize is deprecated since 1.1 and will be removed in 1.2",
                FutureWarning,
            )

        return RidgeClassifierCV(
            alphas=self.alphas,
            fit_intercept=self.fit_intercept,
            scoring=self.scoring,
            normalize=self.normalize,
            cv=self.cv,
            class_weight=self.class_weight,
            store_cv_values=False,
        )

    def predict_proba(self, x):
        decision = self.decision_function(x)
        if decision.ndim == 1:
            decision_2d = np.c_[-decision, decision]
        else:
            decision_2d = decision
        return softmax(decision_2d, copy=False)


class TransformRidgeCV(RegressorMixin, TransformRegressorMixin, BaseTransformEstimator):
    def __init__(
        self,
        *,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        normalize=False,
        scoring=None,
        cv=None,
        gcv_mode=None,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(random_state=random_state, n_jobs=n_jobs)
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.cv = cv
        self.gcv_mode = gcv_mode

    def _get_estimator(self, random_state):
        return RidgeCV(
            alphas=self.alphas,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            scoring=self.scoring,
            cv=self.cv,
            gcv_mode=self.gcv_mode,
            store_cv_values=False,
            alpha_per_target=False,
        )
