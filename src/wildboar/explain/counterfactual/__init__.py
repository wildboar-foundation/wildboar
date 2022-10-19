# Authors: Isak Samsten
# License: BSD 3 clause

from sklearn.utils import deprecated

from ._helper import _proximity as proximity
from ._helper import counterfactuals
from ._nn import KNeighborsCounterfactual
from ._proto import PrototypeCounterfactual
from ._sf import ShapeletForestCounterfactual

__all__ = [
    "counterfactuals",
    "score",
    "proximity",
    "ShapeletForestCounterfactual",
    "KNeighborsCounterfactual",
    "PrototypeCounterfactual",
]


@deprecated(
    "Function 'score' was renamed to 'proximity' in 1.1 and will be removed in 1.2."
)
def score(x_true, x_counterfactuals, metric="euclidean", success=None):
    return proximity(x_true[success], x_counterfactuals[success], metric=metric)
