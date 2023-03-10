# Authors: Isak Samsten
# License: BSD 3 clause

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
    "iseos",  # TODO(1.4): Remove
]


# TODO(1.4): Remove
@deprecated(
    "To be removed in 1.4. Use wildboar.utils.variable_len.is_end_of_series instead."
)
def iseos(x):
    """Boolean indicator if a value is the end-of-sequence value.

    Parameters
    ----------
    x : array-like
        The array

    Returns
    -------
    ndarray
        boolean indicator array
    """
    isneginf = np.isneginf(x)
    if isneginf.any():
        warnings.warn(
            "Using -np.inf as eos has been deprecated in 1.3 and support will "
            "be removed in 1.4. Use wildboar.utils.variable_len.is_end_of_series",
            DeprecationWarning,
        )
    return isneginf or is_end_of_series(x)
