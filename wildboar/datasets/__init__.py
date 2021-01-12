# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten

import os
import re

import numpy as np

from ._repository import ArffBundle, NpyBundle, Bundle, Repository, RepositoryCollection

__all__ = [
    "Repository",
    "Bundle",
    "ArffBundle",
    "NpyBundle",
    "get_bundle",
    "load_dataset",
    "load_all_datasets",
    "load_two_lead_ecg",
    "load_synthetic_control",
    "load_gun_point",
]

_REPOSITORIES = RepositoryCollection()


def _default_cache_dir():
    return _os_cache_path("wildboar")


def _os_cache_path(dir):
    import platform

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


def load_synthetic_control(merge_train_test=True):
    """Load the Synthetic_Control dataset

    See Also
    --------
    load_dataset : load a named dataset
    """
    return load_dataset(
        "SyntheticControl",
        bundle="wildboar/ucr-tiny",
        merge_train_test=merge_train_test,
    )


def load_two_lead_ecg(merge_train_test=True):
    """Load the TwoLeadECG dataset

    See Also
    --------
    load_dataset : load a named dataset
    """
    return load_dataset(
        "TwoLeadECG",
        bundle="wildboar/ucr-tiny",
        merge_train_test=merge_train_test,
    )


def load_gun_point(merge_train_test=True):
    """Load the GunPoint dataset

    See Also
    --------
    load_dataset : load a named dataset
    """
    return load_dataset(
        "GunPoint",
        bundle="wildboar/ucr-tiny",
        merge_train_test=merge_train_test,
    )


def load_all_datasets(
    bundle="wildboar/ucr",
    *,
    cache_dir=None,
    create_cache_dir=True,
    progress=True,
    force=False,
    **kwargs
):
    """Load all datasets as a generator

    Parameters
    ----------
    bundle : str
        A string with the bundle.

    progress : bool, optional
        If progress indicator is shown while downloading the bundle.

    cache_dir : str, optional
        The cache directory for downloaded dataset bundles.

    create_cache_dir : bool, optional
        Create the cache directory if it does not exist.

    force : bool, optional
            Force re-download of cached bundle

    kwargs : dict
        Optional arguments to ``load_dataset``

    Yields
    ------
    x : array-like
        Data samples

    y : array-like
        Data labels

    Examples
    --------

    >>> from wildboar.datasets import load_all_datasets
    >>> for dataset, (x, y) in load_all_datasets(bundle='wildboar/ucr'):
    >>>     print(dataset, x.shape, y.shape)
    """
    for dataset in list_datasets(
        bundle=bundle,
        cache_dir=cache_dir,
        create_cache_dir=create_cache_dir,
        progress=progress,
        force=force,
    ):
        yield dataset, load_dataset(dataset, bundle=bundle, **kwargs)


def load_dataset(
    name,
    *,
    bundle="wildboar/ucr",
    dtype=None,
    contiguous=True,
    merge_train_test=True,
    cache_dir=None,
    create_cache_dir=True,
    progress=True,
    force=False
):
    """Load a dataset from a bundle

    Parameters
    ----------
    name : str
        The name of the dataset to load.

    bundle : str or Bundle, optional
        The data bundle

        - if str load a named bundle, format {repository}/{bundle}
        - if Bundle, load from a given bundle

    dtype : dtype, optional, default=np.float64
        The data type of the returned data

    contiguous : bool, optional
        Ensure that the returned dataset is memory contiguous.

    merge_train_test : bool, optional
        Merge the existing training and testing partitions.

    progress: bool, optional
        Show a progress bar while downloading a bundle.

    cache_dir: str, optional
        The directory where downloaded files are cached

    create_cache_dir: bool, optional
        Create cache directory if missing (default=True)

    force : bool, optional
        Force re-download of already cached bundle

        ..versionadded :: 1.0.4

    Returns
    -------
    x : ndarray
        The data samples

    y : ndarray
        The labels

    x_train : ndarray, optional
        The training samples if ``merge_train_test=False``

    x_test : ndarray, optional
        The testing samples if ``merge_train_test=False``

    y_train : ndarray, optional
        The training labels if ``merge_train_test=False``

    y_test : ndarray, optional
        The testing labels if ``merge_train_test=False``

    Notes
    -----
    A dataset bundle is defined as zip-file. Matching files will be considered as dataset parts.
    Parts sharing the same name will be merged (two files with the same name in different folders share
    name). Filenames (without extension) with the suffix '_TRAIN' or '_TEST' are considered as training
    and testing parts and are used togheter with the attribute ``merge_train_test=False``. Parts without any suffix
    are considered as training parts.

    - `Currently only ".arff" and ".npy" files are supported.`
    - To support other data formats create subclasses of ``Bundle``.

    A repository


    Examples
    --------

    Load one of the bundled datasets

    >>> x, y = load_dataset("synthetic_control")

    or if original training and testing splits are to be preserved

    >>> x_train, x_test, y_train, y_test = load_dataset("synthetic_control", merge_train_test=False)

    and with training and testing parts

    >>> x_train, x_test, y_train, y_test = load_dataset("Wafer", bundle='wildboar/ucr', merge_train_test=False)

    """
    repository, bundle = get_bundle(bundle)
    dtype = dtype or np.float64
    cache_dir = os.path.join(cache_dir or _default_cache_dir(), repository)
    ret_val = []
    x, y, n_train_samples = bundle.load(
        name,
        dtype=dtype,
        cache_dir=cache_dir,
        force=force,
        create_cache_dir=create_cache_dir,
        progress=progress,
    )

    if merge_train_test:
        ret_val.append(x)
        ret_val.append(y)
    else:
        if n_train_samples == x.shape[0]:
            raise ValueError("found no test parts. Set merge_train_test=True.")

        ret_val.append(x[:n_train_samples])
        ret_val.append(x[n_train_samples:])
        ret_val.append(y[:n_train_samples])
        ret_val.append(y[n_train_samples:])

    if contiguous:
        return [np.ascontiguousarray(x) for x in ret_val]
    else:
        return ret_val


def list_datasets(
    bundle="wildboar/ucr",
    *,
    cache_dir=None,
    create_cache_dir=True,
    progress=True,
    force=False
):
    """List the datasets in the bundle

    Parameters
    ----------
    bundle : str or bundle, optional
        The data bundle

        - if `None` load one of the bundled data sets
        - if str load a named bundle
        - if str http(s) or file url, load it as a bundle

    progress: bool, optional
        Show a progress bar while downloading a bundle.

    cache_dir: str, optional
        The directory where downloaded files are cached (default='wildboar_cache')

    create_cache_dir: bool, optional
        Create cache directory if missing (default=True)

    force : bool, optional
        Force re-download of cached bundle

    Returns
    -------
        dataset : set
            A set of dataset names
    """
    repository, bundle = get_bundle(bundle)
    cache_dir = os.path.join(cache_dir or _default_cache_dir(), repository)
    return bundle.list(
        cache_dir=cache_dir,
        create_cache_dir=create_cache_dir,
        progress=progress,
        force=force,
    )


def get_bundle(bundle_name):
    """Get a bundle from a str

    Parameters
    ----------
    bundle_name : str
        Find a named bundle

    Returns
    -------
    repository : str
        Key of the repository

    bundle : Bundle
        A bundle

    """
    repo_bundle_match = re.match("([a-zA-Z]+)/([a-zA-Z0-9\-]+)$", bundle_name)
    if repo_bundle_match:
        repository_name = repo_bundle_match.group(1)
        bundle_name = repo_bundle_match.group(2)
        repository = _REPOSITORIES[repository_name]
        if repository:
            bundle = repository.get_bundle(bundle_name)
            if bundle:
                return repository_name, bundle
            else:
                raise ValueError("bundle (%s) does not exist" % bundle_name)
        else:
            raise ValueError("repository (%s) does not exist" % repository_name)
    else:
        raise ValueError("bundle (%s) is not supported" % bundle_name)


def install_repository(repository):
    """Install repository

    Parameters
    ----------
    repository : str or Repository
        A repository
    """
    if isinstance(repository, str):
        repository = Repository(repository)
    _REPOSITORIES.append(repository)


install_repository(
    "https://raw.githubusercontent.com/isaksamsten/wildboar-datasets/master/repo.json"
)
