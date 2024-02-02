# Authors: Isak Samsten
# License: BSD 3 clause

"""Utilities for variable length series."""
import numpy as np


def _float_to_raw_int_bits(value):
    return np.array(value, dtype="f").view(dtype="I")


def _int_bits_to_float(bits):
    return np.array(bits, dtype="I").view(dtype="f")


eos = EoS = EOS = _int_bits_to_float(0x7F800009).item()
_END_OF_SERIES_MASK = np.array(0x0000000F, dtype=np.uint64)


def is_end_of_series(x):
    """
    Test element-wise for EoS and return result as a boolean array.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    ndarray
        Boolean indicator array.
    """
    return np.logical_and(
        np.isnan(x),
        np.bitwise_and(_float_to_raw_int_bits(x), _END_OF_SERIES_MASK) == 9,
    )


def is_variable_length(x):
    """
    Test if time-series is variable length.

    Parameters
    ----------
    x : time-series
        The input.

    Returns
    -------
    bool
        True if time series contains EoS.
    """
    return is_end_of_series(x).any()


def get_variable_length(x):
    """
    Return the length of each time-series.

    Parameters
    ----------
    x : time-series
        The input.

    Returns
    -------
    float or ndarray
        The lenght of the time series.

    Examples
    --------
    >>> from wildboar.utils.variable_len import get_variable_length
    >>> x = np.array(
    ...     [
    ...         [[1, 2, 3, eos], [1, 2, eos, eos], [1, 2, 3, 4]],
    ...         [[1, 2, 3, eos], [1, 2, eos, eos], [1, eos, 3, 4]],
    ...     ]
    ... )
    >>> get_variable_length(x)
    [[3 2 4]
     [3 2 4]]

    >>> get_variable_length([1, 2, eos, eos])
    2

    >>> get_variable_length([[1, 2, eos, eos]])
    [2]
    """
    eos = is_end_of_series(x)
    if eos.ndim == 1:
        argmax = np.r_[False, eos].argmax() - 1
        return eos.shape[0] if argmax < 0 else argmax
    else:
        if eos.ndim == 2:
            f = np.broadcast_to(False, shape=(eos.shape[0], 1))
        elif eos.ndim == 3:
            f = np.broadcast_to(False, shape=(eos.shape[0], eos.shape[1], 1))
        else:
            raise ValueError(
                f"Found array with {eos.ndim} dimensions but expected <= 3"
            )

        argmax = np.c_[f, eos].argmax(axis=-1) - 1
        argmax[argmax == -1] = eos.shape[-1]
        return argmax
