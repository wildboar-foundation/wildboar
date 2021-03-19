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
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, class_weight
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.linear_model import RidgeClassifierCV

from ..embed import RocketEmbedding


class RocketClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_kernels=10000,
        *,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        normalize=False,
        scoring=None,
        cv=None,
        class_weight=None,
        random_state=None
    ):
        self.n_kernels = n_kernels
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.cv = cv
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, x, y, sample_weight=None):
        x = check_array(x, dtype=np.float64, order="C")
        y = check_array(y, ensure_2d=False, order="C")
        random_state = check_random_state(self.random_state)
        self.pipe_ = make_pipeline(
            RocketEmbedding(self.n_kernels, random_state=random_state),
            RidgeClassifierCV(
                alphas=self.alphas,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                scoring=self.scoring,
                cv=self.cv,
                class_weight=self.class_weight,
                store_cv_values=False,
            ),
        )
        self.pipe_.fit(x, y, ridgeclassifiercv__sample_weight=sample_weight)
        return self

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
