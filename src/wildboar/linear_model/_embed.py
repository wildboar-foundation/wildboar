# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state


class BaseEmbeddingEstimator(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, *, random_state=None, n_jobs=None):
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, x, y, sample_weight=None):
        x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions")

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
        x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions")
        return self.pipe_.predict(x)

    def predict_proba(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions")
        return self.pipe_.predict_proba(x)

    def predict_log_proba(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions")
        return self.pipe_.predict_log_proba(x)

    def decision_function(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions")
        return self.pipe_.decision_function(x)


class EmbeddingRegressorMixin:
    def predict(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions")
        return self.pipe_.predict(x)

    def decision_function(self, x):
        check_is_fitted(self)
        x = check_array(x, dtype=np.float64, allow_nd=True, order="C")
        if x.ndim < 2 or x.ndim > 3:
            raise ValueError("illegal input dimensions")
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
