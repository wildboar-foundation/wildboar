"""Ensemble methods for classification, regression and outlier detection."""

# Authors: Isak Samsten
# License: BSD 3 clause

from ._elastic import ElasticEnsembleClassifier
from ._ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    BaseBagging,
    ExtraShapeletTreesClassifier,
    ExtraShapeletTreesRegressor,
    IntervalForestClassifier,
    IntervalForestRegressor,
    IsolationShapeletForest,
    PivotForestClassifier,
    ProximityForestClassifier,
    RocketForestClassifier,
    RocketForestRegressor,
    ShapeletForestClassifier,
    ShapeletForestEmbedding,
    ShapeletForestRegressor,
)

__all__ = [
    "BaseBagging",
    "BaggingClassifier",
    "BaggingRegressor",
    "ShapeletForestClassifier",
    "ExtraShapeletTreesClassifier",
    "ShapeletForestRegressor",
    "ExtraShapeletTreesRegressor",
    "IsolationShapeletForest",
    "ShapeletForestEmbedding",
    "RocketForestClassifier",
    "RocketForestRegressor",
    "IntervalForestClassifier",
    "IntervalForestRegressor",
    "PivotForestClassifier",
    "ProximityForestClassifier",
    "ElasticEnsembleClassifier",
]
