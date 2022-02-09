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
# Authors: Isak Samsten

from functools import wraps

import numpy as np

__all__ = [
    "array_or_scalar",
    "singleton",
]


def _array_or_scalar(x, squeeze=True):
    if not isinstance(x, np.ndarray):
        return x

    if x.size == 1:
        return x.item()
    else:
        return np.squeeze(x) if squeeze else x


def array_or_scalar(optional_f=None, squeeze=True):
    """Decorate a function returning an ndarray to return a single scalar if the array
    has a single item.

    Parameters
    ----------
    optional_f : callable, optional
        Used if the decorator is used without arguments

    squeeze : bool, optional
        Remove axis of length one from the returned arrays
    """

    def decorator(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            x = f(*args, **kwargs)
            if isinstance(x, tuple):
                return tuple(_array_or_scalar(r, squeeze=squeeze) for r in x)
            elif isinstance(x, np.ndarray):
                return _array_or_scalar(x, squeeze=squeeze)
            else:
                return x

        return wrap

    return decorator(optional_f) if callable(optional_f) else decorator


def _singleton(x):
    if isinstance(x, list):
        return _singleton(x[0]) if len(x) == 1 else x
    else:
        return x


def singleton(f):
    """Recursivley try to unwrap list return arguments such that a single element can
    be returned
    """

    @wraps(f)
    def wrap(*args, **kwargs):
        ret_vals = f(*args, **kwargs)
        if isinstance(ret_vals, tuple):
            return tuple(_singleton(ret_val) for ret_val in ret_vals)
        else:
            return _singleton(ret_vals)

    return wrap


def unstable(optional_f=None, stability="beta", description=None):
    """Decorate a function as unsatable

    Parameters
    ----------

    optional_f : callable, optional
        The decorated function

    stability : str, optional
        The stability of the feature

    description : str, optional
        The description of the feature

    """
    import warnings

    def decorator(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            warning = f"'{f.__qualname__}' is currently in '{stability}'."
            if description is not None:
                warning += f" {description}."
            warnings.warn(warning, FutureWarning)
            return f(*args, **kwargs)

        return wrap

    return decorator(optional_f) if callable(optional_f) else decorator
