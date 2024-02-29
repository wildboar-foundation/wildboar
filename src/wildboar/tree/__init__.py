# Authors: Isak Samsten
# License: BSD 3 clause

"""Tree-based estimators for classification and regression."""

from ._plot import plot_tree
from ._ptree import ProximityTreeClassifier
from ._tree import (
    ExtraShapeletTreeClassifier,
    ExtraShapeletTreeRegressor,
    IntervalTreeClassifier,
    IntervalTreeRegressor,
    PivotTreeClassifier,
    RocketTreeClassifier,
    RocketTreeRegressor,
    ShapeletTreeClassifier,
    ShapeletTreeRegressor,
)

__all__ = [
    "ShapeletTreeClassifier",
    "ShapeletTreeRegressor",
    "ExtraShapeletTreeClassifier",
    "ExtraShapeletTreeRegressor",
    "RocketTreeClassifier",
    "RocketTreeRegressor",
    "IntervalTreeClassifier",
    "IntervalTreeRegressor",
    "PivotTreeClassifier",
    "ProximityTreeClassifier",
    "plot_tree",
]
