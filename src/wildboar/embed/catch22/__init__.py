# cython: language_level=3

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

from functools import partial

from sklearn.utils import check_array

from . import _catch22

__all__ = ["histogram_mode", "histogram_mode5", "histogram_mode10"]


def histogram_mode(x, n_bins=5):
    """Compute the histogram mode

    Parameters
    ----------
    x : ndarray of shape (n_timestep, )
       The input array

    n_bins : int, optional
       The number of bins

    Returns
    -------
    mode : float
       The histogram mode
    """
    if x.ndim != 1:
        raise ValueError("1d required")

    x = check_array(x, ensure_2d=False, order="c")
    return _catch22.histogram_mode(x, n_bins)


histogram_mode5 = partial(histogram_mode, n_bins=5)


histogram_mode10 = partial(histogram_mode, n_bins=10)
