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

import os
import platform
import warnings

from wildboar.utils.data import check_dataset

from ._validation import check_array, check_X_y

__all__ = [
    "check_array",
    "check_X_y",
    "check_dataset",
    "os_cache_dir",
    "_soft_dependency_error",
    "DependencyMissing",
]


class DependencyMissing:
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
    """Get the path to a operating system specific cache directory

    Parameters
    ----------
    dir : str
        The sub-directory in the cache location

    Returns
    -------
    path : str
        The cache path
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
        warnings.warn(msg)
    else:
        if e is not None:
            raise ModuleNotFoundError(msg) from e
        else:
            raise ModuleNotFoundError(msg)
