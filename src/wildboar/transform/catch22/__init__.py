# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

from functools import partial

from ...utils.decorators import array_or_scalar
from ...utils.validation import check_array
from . import _catch22

__all__ = ["histogram_mode", "histogram_mode5", "histogram_mode10"]


@array_or_scalar
def histogram_mode(x, n_bins=5):
    """Compute the histogram mode

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep) or (n_timestep, )
       The input array

    n_bins : int, optional
       The number of bins

    Returns
    -------
    mode : array or float
       The histogram mode
    """
    x = check_array(x, allow_3d=True)
    return _catch22.histogram_mode_(x, n_bins)


histogram_mode5 = partial(histogram_mode, n_bins=5)
histogram_mode5.__doc__ = histogram_mode.__doc__


histogram_mode10 = partial(histogram_mode, n_bins=10)
histogram_mode5.__doc__ = histogram_mode.__doc__
