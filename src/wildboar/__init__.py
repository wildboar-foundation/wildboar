# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten
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
