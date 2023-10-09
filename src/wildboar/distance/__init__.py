"""
Fast distance computations.

The :py:mod:`wildboar.distance` module includes functions for computing
paired and pairwise distances between time series and between time series and
subsequences.

See the :ref:`User Guide <guide-metrics>` for more details and
examples.
"""

# Authors: Isak Samsten
# License: BSD 3 clause

from ._distance import (
    _METRICS,  # noqa: F401
    _SUBSEQUENCE_METRICS,  # noqa: F401
    mean_paired_distance,  # Deprecated
    paired_distance,
    paired_subsequence_distance,
    paired_subsequence_match,
    pairwise_distance,
    pairwise_subsequence_distance,
    subsequence_match,
)
from ._matrix_profile import matrix_profile
from ._neighbour import KNeighbourClassifier

__all__ = [
    "pairwise_subsequence_distance",
    "paired_subsequence_distance",
    "subsequence_match",
    "paired_subsequence_match",
    "pairwise_distance",
    "paired_distance",
    "matrix_profile",
    "KNeighbourClassifier",
    "mean_paired_distance",  # Deprecated
]
