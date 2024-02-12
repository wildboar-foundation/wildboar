# Authors: Isak Samsten
# License: BSD 3 clause

"""
Wildboar - a library for temporal machine learning.

Wildboar includes numerous temporal machine learning algorithms and seamlessly
integrates them with `scikit-learn <https://scikit-learn.org>`__.
"""
from .utils.variable_len import EOS, EoS, eos
from .version import version as __version__

__all__ = [
    "__version__",
    "eos",
    "EoS",
    "EOS",
]
