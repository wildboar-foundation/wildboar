# Authors: Isak Samsten
# License: BSD 3 clause

"""Utility functions."""

import os
import platform
import warnings

import numpy as np

from .validation import check_array, check_X_y

__all__ = [
    "check_array",
    "check_X_y",
]


class DependencyMissing:
    """
    Class to flag dependencies as missing.

    Parameters
    ----------
    e : ModuleNotFoundError, optional
        An exception.
    package : str, optional
        The package (installable using `pip`) required for the
        desired functionality.
    context : str, optional
        The context in which the error was raised.

    Examples
    --------
    In the following example we assume that `matplotlib` has not been installed
    by the user.

    >>> import numpy as np
    >>> from wildboar.utils import DependencyMissing
    >>> try:
    ...     import matplotlib.pylab as plt
    ... except ModuleNotFoundError as e:
    ...     plt = DependencyMissing(e, package="matplotlib", context="plot")
    >>> plt.plot(np.arange(10))
    Traceback (most recent call last):
    File "<stdin>", line 2, in <module>
    ModuleNotFoundError: No module named 'matplotlib'

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
    ....
    ModuleNotFoundError: 'matplotlib' is required for 'plot', but not included
    in the default wildboar installation. Please run: `pip install matplotlib`
    to install the required package.
    """

    def __init__(self, e=None, *, package=None, context=None):
        if context is None:
            import inspect

            frm = inspect.stack()[1]
            if frm.function == "<module>":
                frm = inspect.getmodule(frm.frame)
                context = frm.__name__ if frm is not None else ""
            else:
                context = frm.function
        self.__package = package or e.name
        self.__context = context
        self.__e = e

    def __mro_entries__(self, bases):
        class Mock:
            def __init__(ignore_me, *args, **kwargs):
                _soft_dependency_error(
                    e=self.__e, package=self.__package, context=self.__context
                )

        return (Mock,)

    def __getattr__(self, name):
        _soft_dependency_error(
            e=self.__e, package=self.__package, context=self.__context
        )

    def __call__(self, *args, **kwds):
        _soft_dependency_error(
            e=self.__e, package=self.__package, context=self.__context
        )


def os_cache_path(dir):
    """
    Get the path to a operating system specific cache directory.

    Parameters
    ----------
    dir : str
        The sub-directory in the cache location.

    Returns
    -------
    str
        The cache path.
    """
    if platform.system() == "Windows":
        cache_dir = os.path.expandvars(r"%LOCALAPPDATA%\cache")
        return os.path.join(cache_dir, dir)
    elif platform.system() == "Linux":
        cache_dir = os.environ.get("XDG_CACHE_HOME")
        if not cache_dir:
            cache_dir = ".cache"
        return os.path.join(os.path.expanduser("~"), cache_dir, dir)
    elif platform.system() == "Darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Caches", dir)
    else:
        return os.path.join(os.path.expanduser("~"), ".cache", dir)


def _soft_dependency_error(e=None, package=None, context=None, warning=False):
    if e is None and package is None:
        raise ValueError("both e and package cant be None")

    if context is None:
        import inspect

        frm = inspect.stack()[1]
        if frm.function == "<module>":
            context = inspect.getmodule(frm.frame).__name__
        else:
            context = frm.function
    package = package or e.name
    msg = (
        f"'{package}' is required for '{context}', but not included in the default "
        f"wildboar installation. Please run: `pip install {package}` to install the "
        f"required package."
    )
    if warning:
        warnings.warn(msg, UserWarning)
    elif e is not None:
        raise ModuleNotFoundError(msg) from e
    else:
        raise ModuleNotFoundError(msg)


def _safe_jagged_array(lst):
    arr = np.empty(len(lst), dtype=object)
    arr[:] = lst
    return arr
