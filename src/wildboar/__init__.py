# Authors: Isak Samsten
# License: BSD 3 clause

"""
Wildboar - a library for temporal machine learning.

Wildboar includes numerous temporal machine learning algorithms and seamlessly
integrates them with `scikit-learn <https://scikit-learn.org>`__.
"""

import warnings

import numpy as np
from sklearn.utils.deprecation import deprecated

from .utils.variable_len import EOS, EoS, eos, is_end_of_series
from .version import version as __version__

__all__ = [
    "__version__",
    "eos",
    "EoS",
    "EOS",
    "iseos",  # TODO(1.3): Remove
]


# TODO(1.3): Remove
@deprecated(
    "To be removed in 1.3. Use wildboar.utils.variable_len.is_end_of_series instead."
)
def iseos(x):
    """
    Boolean indicator if a value is the end-of-sequence value.

    Parameters
    ----------
    x : array-like
        The array.

    Returns
    -------
    ndarray
        Boolean indicator array.
    """
    isneginf = np.isneginf(x)
    if isneginf.any():
        # TODO(1.3)
        warnings.warn(
            "Using -np.inf as eos has been deprecated in 1.2 and support will "
            "be removed in 1.3. Use wildboar.utils.variable_len.is_end_of_series",
            DeprecationWarning,
        )
        return np.logical_or(isneginf, is_end_of_series(x))

    return is_end_of_series(x)
