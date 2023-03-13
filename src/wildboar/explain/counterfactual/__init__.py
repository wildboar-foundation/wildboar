"""Counterfactual explanations.

The :mod:`wildboar.explain.counterfactual` module includes numerous
methods for generating counterfactual explanations.
"""
# Authors: Isak Samsten
# License: BSD 3 clause

from ._helper import _proximity as proximity
from ._helper import counterfactuals
from ._nn import KNeighborsCounterfactual
from ._proto import PrototypeCounterfactual
from ._sf import ShapeletForestCounterfactual

__all__ = [
    "counterfactuals",
    "proximity",
    "ShapeletForestCounterfactual",
    "KNeighborsCounterfactual",
    "PrototypeCounterfactual",
]
