"""Linear methods for both classification and regression."""

# Authors: Isak Samsten
# License: BSD 3 clause

from ._hydra import HydraClassifier
from ._rocket import RocketClassifier, RocketRegressor
from ._shapelet import (
    CastorClassifier,
    DilatedShapeletClassifier,
    RandomShapeletClassifier,
    RandomShapeletRegressor,
)

__all__ = [
    "HydraClassifier",
    "RocketClassifier",
    "RocketRegressor",
    "RandomShapeletClassifier",
    "RandomShapeletRegressor",
    "DilatedShapeletClassifier",
    "CastorClassifier",
]
