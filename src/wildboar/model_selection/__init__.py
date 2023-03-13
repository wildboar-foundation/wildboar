"""Methods for model selection."""
# Authors: Isak Samsten
# License: BSD 3 clause

from ._cv import RepeatedOutlierSplit
from ._outlier import outlier_train_test_split

__all__ = [
    "RepeatedOutlierSplit",
    "outlier_train_test_split",
]
