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
from sklearn.utils import deprecated

from ._repository import (
    ArffBundle,
    NpyBundle,
    Bundle,
    Repository,
    RepositoryCollection,
    JSONRepository,
)

from ._filter import make_filter

__all__ = [
    "Repository",
    "JSONRepository",
    "Bundle",
    "ArffBundle",
    "NpyBundle",
    "set_cache_dir",
    "get_repository",
    "clear_cache",
    "list_repositories",
    "list_bundles",
    "get_bundles",
    "install_repository",
    "load_dataset",
    "list_datasets",
    "load_datasets",
    "load_all_datasets",
    "load_two_lead_ecg",
    "load_synthetic_control",
    "load_gun_point",
]

_CACHE_DIR = None
_REPOSITORIES = RepositoryCollection()


def _split_repo_bundle(repo_bundle_name):
    """Split a repository bundle string of the format {repo}/{bundle}

    Parameters
    ----------
    repo_bundle_name : str
        {repo}/{bundle} name

    Returns
    -------
    repository : str
        Key of the repository

    bundle : str
        Name of the bundle

    version : str
        An optional version

    tag : str
        An optional tag
    """
    match = re.match(
        "(?:([a-zA-Z]+)/([a-zA-Z0-9\-]+))?(?::((?:\d+\.)?(?:\d+\.)?(?:\*|\d+)))?(?::([a-zA-Z\-]+))?$",
        repo_bundle_name,
    )
    if match:
        repository_name = match.group(1)
        bundle_name = match.group(2)
        version = match.group(3)
        if version:
            version = re.sub("(\d+\.\d+)((?:\.0+)*)$", "\\1", version)
        tag = match.group(4)
        return repository_name, bundle_name, version, tag
    else:
        raise ValueError("repository (%s) is not supported" % repo_bundle_name)


def set_cache_dir(cache_dir):
    """Change the global cache directory

    cache_dir : str
        The cache directory root
    """
    global _CACHE_DIR
    _CACHE_DIR = cache_dir


def _default_cache_dir():
    return _os_cache_path("wildboar") if not _CACHE_DIR else _CACHE_DIR


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
        repository="wildboar/ucr-tiny",
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
        repository="wildboar/ucr-tiny",
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
        repository="wildboar/ucr-tiny",
        merge_train_test=merge_train_test,
    )


def load_datasets(
    repository="wildboar/ucr",
    *,
    cache_dir=None,
    create_cache_dir=True,
    progress=True,
    force=False,
    filter=None,
    **kwargs
):
    """Load all datasets as a generator

    Parameters
    ----------
    repository : str
        The repository string

    progress : bool, optional
        If progress indicator is shown while downloading the repository.

    cache_dir : str, optional
        The cache directory for downloaded dataset repositories.

    create_cache_dir : bool, optional
        Create the cache directory if it does not exist.

    force : bool, optional
        Force re-download of cached repository

    filter: dict or callable, optional
        Filter the datasets

        - if callable, only yield those datasets for which the callable returns True.
          ``f(dataset, x, y) -> bool``

        - if dict, filter based on the keys and values
            - "dataset": regex matching dataset name
            - "n_samples": comparison spec
            - "n_timestep": comparison spec

        Comparison spec
            str of two parts, comparison operator (<, <=, >, >= or =) and a number, e.g., "<100", "<= 200", or ">300"

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

    >>> from wildboar.datasets import load_datasets
    >>> for dataset, (x, y) in load_datasets(repository='wildboar/ucr'):
    >>>     print(dataset, x.shape, y.shape)

    Print the names of datasets with more than 200 samples

    >>> for dataset, (x, y) in load_datasets(repository='wildboar/ucr', filter={"n_samples": ">200"}):
    >>>     print(dataset)
    """
    for dataset in list_datasets(
        repository=repository,
        cache_dir=cache_dir,
        create_cache_dir=create_cache_dir,
        progress=progress,
        force=force,
    ):
        x, y = load_dataset(dataset, repository=repository, **kwargs)
        if filter is None:
            yield dataset, (x, y)
        elif hasattr(filter, "__call__"):
            if filter(dataset, x, y):
                yield dataset, (x, y)
        else:
            if make_filter(filter)(dataset, x, y):
                yield dataset, (x, y)


@deprecated(
    "the function datasets.load_all_datasets has been deprecated in "
    "1.0.3 and will be removed in in 1.0.5."
)
def load_all_datasets(
    repository="wildboar/ucr",
    *,
    cache_dir=None,
    create_cache_dir=True,
    progress=True,
    force=False,
    **kwargs
):
    return load_datasets(
        repository,
        cache_dir=cache_dir,
        create_cache_dir=create_cache_dir,
        progress=progress,
        force=force,
        **kwargs
    )


def load_dataset(
    name,
    *,
    repository="wildboar/ucr",
    dtype=None,
    contiguous=True,
    merge_train_test=True,
    cache_dir=None,
    create_cache_dir=True,
    progress=True,
    force=False
):
    """Load a dataset from a repository

    Parameters
    ----------
    name : str
        The name of the dataset to load.

    repository : str, optional
        The data repository formatted as {repository}/{bundle}[:{version}][:{tag}]

    dtype : dtype, optional
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

    Examples
    --------

    Load a dataset from the default repository

    >>> x, y = load_dataset("SyntheticControl")

    or if original training and testing splits are to be preserved

    >>> x_train, x_test, y_train, y_test = load_dataset("SyntheticControl", merge_train_test=False)

    or for a specific version of the dataset

    >>> x_train, x_test, y_train, y_test = load_dataset("Wafer", repository='wildboar/ucr-tiny:1.0')
    """
    (
        repository_name,
        bundle_name,
        bundle_version,
        bundle_tag,
    ) = _split_repo_bundle(repository)
    dtype = dtype or np.float64
    cache_dir = cache_dir or _default_cache_dir()
    repository = get_repository(repository_name)
    ret_val = []
    x, y, n_train_samples = repository.load_dataset(
        bundle_name,
        name,
        version=bundle_version,
        tag=bundle_tag,
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
    repository="wildboar/ucr",
    *,
    cache_dir=None,
    create_cache_dir=True,
    progress=True,
    force=False
):
    """List the datasets in the repository

    Parameters
    ----------
    repository : str or Bundle, optional
        The data repository

        - if str load a named bundle, format {repository}/{bundle}

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
    (
        repository_name,
        bundle_name,
        bundle_version,
        bundle_tag,
    ) = _split_repo_bundle(repository)
    cache_dir = cache_dir or _default_cache_dir()
    repository = get_repository(repository_name)
    return repository.list_datasets(
        bundle_name,
        version=bundle_version,
        tag=bundle_tag,
        cache_dir=cache_dir,
        create_cache_dir=create_cache_dir,
        progress=progress,
        force=force,
    )


def clear_cache(repository=None, *, cache_dir=None, keep_last_version=True):
    """Clear the cache by deleting cached datasets

    Parameters
    ----------
    repository : str, optional
        The name of the repository to clear cache.

        - if None, clear cache of all repositories

    cache_dir : str, optional
        The cache directory

    keep_last_version : bool, optional
        If true, keep the latest version of each repository.
    """
    cache_dir = cache_dir or _default_cache_dir()
    if repository is None:
        for repo in _REPOSITORIES:
            repo.clear_cache(cache_dir=cache_dir, keep_last_version=keep_last_version)
    else:
        get_repository(repository).clear_cache(
            cache_dir=cache_dir, keep_last_version=keep_last_version
        )


def get_repository(repository):
    """Get repository by name

    Parameters
    ----------
    repository : str
        Repository name

    Returns
    -------
    repository : Repository
        A repository
    """
    if repository in _REPOSITORIES:
        return _REPOSITORIES[repository]
    else:
        raise ValueError("repository (%s) does not exist" % repository)


def install_repository(repository):
    """Install repository

    Parameters
    ----------
    repository : str or Repository
        A repository
    """
    if isinstance(repository, str):
        repository = JSONRepository(repository)
    _REPOSITORIES.append(repository)


def get_bundles(repository):
    """Get all bundles in the repository

    Parameters
    ----------
    repository : str
        Name of the repository

    Returns
    -------
    dict : A dict of key Bundle pairs
    """
    if repository in _REPOSITORIES:
        return _REPOSITORIES[repository].get_bundles()
    else:
        raise ValueError("repository (%s) does not exist.")


def list_bundles(repository):
    """Get a list of all bundle names in the specified repository.

    Parameters
    ----------
    repository : str
        The name of the repository

    Returns
    -------
    bundle : str
        The name of the bundle
    """
    return [key for key, bundle in get_bundles(repository).items()]


def list_repositories():
    """List the key of all installed repositories"""
    return [repo.name for repo in _REPOSITORIES]


# Install the default 'wildboar' repository
install_repository("https://isaksamsten.github.io/wildboar-datasets/1.0/repo.json")
