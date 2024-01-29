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
    argmin_distance,
    argmin_subsequence_distance,
    check_metric,  # noqa: F401
    check_subsequence_metric,  # noqa: F401
    distance_profile,
    paired_distance,
    paired_subsequence_distance,
    paired_subsequence_match,
    pairwise_distance,
    pairwise_subsequence_distance,
    subsequence_match,
)
from ._manifold import MDS
from ._matrix_profile import matrix_profile
from ._neighbors import KMeans, KMedoids, KNeighborsClassifier

__all__ = [
    "argmin_distance",
    "argmin_subsequence_distance",
    "distance_profile",
    "pairwise_subsequence_distance",
    "paired_subsequence_distance",
    "subsequence_match",
    "paired_subsequence_match",
    "pairwise_distance",
    "paired_distance",
    "matrix_profile",
    "KNeighborsClassifier",
    "KMeans",
    "KMedoids",
    "MDS",
]
