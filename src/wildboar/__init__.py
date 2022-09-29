# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from .version import version as __version__

__all__ = [
    "__version__",
    "eos",
    "iseos",
]

eos = -np.inf
"""The end-of-sequence identifier. Indicates the end of a time series."""


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
    return np.isneginf(x)
