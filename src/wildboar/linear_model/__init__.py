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

from ._kernel_logistic import KernelLogisticRegression
from ._rocket import RocketClassifier, RocketRegressor
from ._shapelet import RandomShapeletClassifier, RandomShapeletRegressor
from ._base import (
    EmbeddingClassifierMixin,
    EmbeddingRegressorMixin,
    BaseEmbeddingEstimator,
)

__all__ = [
    "KernelLogisticRegression",
    "EmbeddingClassifierMixin",
    "EmbeddingRegressorMixin",
    "BaseEmbeddingEstimator",
    "RocketClassifier",
    "RocketRegressor",
    "RandomShapeletClassifier",
    "RandomShapeletRegressor",
]
