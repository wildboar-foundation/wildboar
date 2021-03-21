from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.random import rand
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state
from sklearn.linear_model import RidgeClassifierCV, RidgeCV

from wildboar.embed.base import BaseEmbedding


class BaseEmbeddingEstimator(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, x, y, sample_weight=None):
        x = check_array(x, dtype=np.float64, order="C")
        y = check_array(y, ensure_2d=False, order="C")
        random_state = check_random_state(self.random_state)
        self.pipe_ = Pipeline(
            [
                ("embedding", self._get_embedding(random_state.randint(2 ** 31))),
                ("estimator", self._get_estimator(random_state.randint(2 ** 31))),
            ],
        )
        self.pipe_.fit(x, y, estimator__sample_weight=sample_weight)
        return self

    @abstractmethod
    def _get_embedding(self, random_state):
        pass

    @abstractmethod
    def _get_estimator(self, random_state):
        pass


class EmbeddingClassifierMixin:
    def predict(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, order="C")
        return self.pipe_.predict(x)

    def predict_proba(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, order="C")
        return self.pipe_.predict_proba(x)

    def predict_log_proba(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, order="C")
        return self.pipe_.predict_log_proba(x)

    def decision_function(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, order="C")
        return self.pipe_.decision_function(x)


class EmbeddingRegressorMixin:
    def predict(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, order="C")
        return self.pipe_.predict(x)

    def decision_function(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, order="C")
        return self.pipe_.decision_function(x)


class EmbeddingRidgeClassifierCV(
    ClassifierMixin, EmbeddingClassifierMixin, BaseEmbeddingEstimator
):
    def __init__(
        self,
        *,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        normalize=False,
        scoring=None,
        cv=None,
        class_weight=None,
        random_state=None,
    ):
        super().__init__(random_state=random_state)
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.cv = cv
        self.class_weight = class_weight
        self.random_state = random_state

    def _get_estimator(self, random_state):
        return RidgeClassifierCV(
            alphas=self.alphas,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            scoring=self.scoring,
            cv=self.cv,
            class_weight=self.class_weight,
            store_cv_values=False,
        )


class EmbeddingRidgeCV(RegressorMixin, EmbeddingRegressorMixin, BaseEmbeddingEstimator):
    def __init__(
        self,
        *,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        normalize=False,
        scoring=None,
        cv=None,
        gcv_mode=None,
        random_state=None,
    ):
        super().__init__(random_state=random_state)
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
