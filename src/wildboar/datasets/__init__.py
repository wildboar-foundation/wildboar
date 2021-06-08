# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
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

from ._repository import (
    Bundle,
    JSONRepository,
    NpBundle,
    Repository,
    RepositoryCollection,
)
from .filter import make_filter
from .preprocess import named_preprocess

__all__ = [
    "Repository",
    "JSONRepository",
    "Bundle",
    "NpBundle",
    "set_cache_dir",
    "get_repository",
    "clear_cache",
    "list_repositories",
    "refresh_repositories",
    "list_bundles",
    "get_bundles",
    "install_repository",
    "load_dataset",
    "list_datasets",
    "load_datasets",
    "list_collections",
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
        r"(?:([a-zA-Z]+)/([a-zA-Z0-9\-]+))?(?::((?:\d+\.)?(?:\d+\.)?(?:\*|\d+)))?(?::([a-zA-Z\-]+))?$",  # noqa: E501
        repo_bundle_name,
    )
    if match:
        repository_name = match.group(1)
        bundle_name = match.group(2)
        version = match.group(3)
        if version:
            version = re.sub(r"(\d+\.\d+)((?:\.0+)*)$", "\\1", version)
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
    collection=None,
    cache_dir=None,
    create_cache_dir=True,
    progress=True,
    force=False,
    filter=None,
    **kwargs,
):
    """Load all datasets as a generator

    Parameters
    ----------
    repository : str
        The repository string

    collection : str, optional
        A collection of named datasets.

    progress : bool, optional
        If progress indicator is shown while downloading the repository.

    cache_dir : str, optional
        The cache directory for downloaded dataset repositories.

    create_cache_dir : bool, optional
        Create the cache directory if it does not exist.

    force : bool, optional
        Force re-download of cached repository

    filter: str, dict, list or callable, optional
        Filter the datasets

        - if callable, only yield those datasets for which the callable returns True.
          ``f(dataset, x, y) -> bool``

        - if dict, filter based on the keys and values, where keys are attributes and
          values comparison specs

        - if list, filter based on conjunction of attribute comparisons

        - if str, filter based on attribute comparison

        Attribute comparison
        --------------------

            The format of attribute comparisons are ``[attribute][comparison spec].

            Valid attributes are
            - ``dataset``
            - ``n_samples``
            - ``n_timestep``
            - ``n_dims``
            - ``n_labels``

        Comparison spec
        ---------------
            str of two parts, comparison operator (<, <=, >, >= or =) and a number,
            e.g., "<100", "<= 200", or ">300"

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

    >>> for dataset, (x, y) in load_datasets(
    ...    repository='wildboar/ucr', filter={"n_samples": ">200"}
    ... ):
    >>>     print(dataset)

    >>> for dataset, (x, y) in load_datasets(
    ...    repository='wildboar/ucr', filter="n_samples>200"
    ... ):
    >>>     print(dataset)
    """
    for dataset in list_datasets(
        repository=repository,
        collection=collection,
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


def load_dataset(
    name,
    *,
    repository="wildboar/ucr",
    dtype=None,
    preprocess=None,
    contiguous=True,
    merge_train_test=True,
    cache_dir=None,
    create_cache_dir=True,
    progress=True,
    return_extras=False,
    force=False,
    refresh=False,
    timeout=None,
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

    preprocess : str or callable, optional
        Preprocess the dataset

        - if str, use named preprocess function (see ``preprocess._PREPROCESS.keys()``
          for valid keys)
        - if callable, function taking a np.ndarray and returns the preprocessed dataset

    merge_train_test : bool, optional
        Merge the existing training and testing partitions.

    progress: bool, optional
        Show a progress bar while downloading a bundle.

    cache_dir: str, optional
        The directory where downloaded files are cached

    create_cache_dir: bool, optional
        Create cache directory if missing (default=True)

    return_extras : bool, optional
        Return optional extras

        ..versionadded :: 1.1

    force : bool, optional
        Force re-download of already cached bundle

        ..versionadded :: 1.0.4

    refresh : bool, optional
        Refresh the repository

        ..versionadded :: 1.1

    timeout : float, optional
        Timeout for json request

        ..versionadded :: 1.1

    Returns
    -------
    x : ndarray
        The data samples, optional

    y : ndarray, optional
        The labels

    x_train : ndarray, optional
        The training samples if ``merge_train_test=False``

    x_test : ndarray, optional
        The testing samples if ``merge_train_test=False``

    y_train : ndarray, optional
        The training labels if ``merge_train_test=False``

    y_test : ndarray, optional
        The testing labels if ``merge_train_test=False``

    extras : dict, optional
        The optional extras if ``return_extras=True``

    Examples
    --------

    Load a dataset from the default repository

    >>> x, y = load_dataset("SyntheticControl")

    or if original training and testing splits are to be preserved

    >>> x_train, x_test, y_train, y_test = load_dataset(
    ...     "SyntheticControl", merge_train_test=False
    ... )

    or for a specific version of the dataset

    >>> x_train, x_test, y_train, y_test = load_dataset(
    ...     "Wafer", repository='wildboar/ucr-tiny:1.0'
    ... )
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

    if refresh:
        repository.refresh(timeout)

    ret_val = []
    x, y, n_train_samples, extras = repository.load_dataset(
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
    if preprocess is None:

        def preprocess(identity):
            return identity

    elif isinstance(preprocess, str):
        preprocess = named_preprocess(preprocess)
    elif not hasattr(preprocess, "__call__"):
        raise ValueError("preprocess (%r) is not supported" % preprocess)

    if merge_train_test:
        ret_val.append(preprocess(x))
        ret_val.append(y)
    else:
        if n_train_samples == x.shape[0]:
            raise ValueError("found no test parts. Set merge_train_test=True.")

        ret_val.append(preprocess(x[:n_train_samples]))
        ret_val.append(preprocess(x[n_train_samples:]))
        if y is not None:
            ret_val.append(y[:n_train_samples])
            ret_val.append(y[n_train_samples:])
        else:
            ret_val.append(None)
            ret_val.append(None)
    if return_extras:
        ret_val.append(extras)

    if contiguous:
        return [
            np.ascontiguousarray(x)
            if isinstance(x, np.ndarray)
            else {k: np.ascontiguousarray(v) for k, v in extras.items()}
            for x in ret_val
        ]
    else:
        return ret_val


def list_datasets(
    repository="wildboar/ucr",
    *,
    collection=None,
    cache_dir=None,
    create_cache_dir=True,
    progress=True,
    force=False,
    refresh=False,
    timeout=None,
):
    """List the datasets in the repository

    Parameters
    ----------
    repository : str or Bundle, optional
        The data repository

        - if str load a named bundle, format {repository}/{bundle}

    collection : str, optional
        A collection of named datasets.

    progress: bool, optional
        Show a progress bar while downloading a bundle.

    cache_dir: str, optional
        The directory where downloaded files are cached (default='wildboar_cache')

    create_cache_dir: bool, optional
        Create cache directory if missing (default=True)

    force : bool, optional
        Force re-download of cached bundle

    refresh : bool, optional
        Refresh the repository

        ..versionadded :: 1.1

    timeout : float, optional
        Timeout for json request

        ..versionadded :: 1.1

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
    if refresh:
        repository.refresh(timeout)
    return repository.list_datasets(
        bundle_name,
        collection=collection,
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
    return _REPOSITORIES[repository]


def install_repository(repository, *, refresh=True, timeout=None):
    """Install repository

    Parameters
    ----------
    repository : str or Repository
        A repository

    refresh : bool, optional
        Refresh the repository

        ..versionadded :: 1.1

    timeout : float, optional
        Timeout for json request

        ..versionadded :: 1.1
    """
    if isinstance(repository, str):
        repository = JSONRepository(repository)
    _REPOSITORIES.append(repository, refresh=refresh, timeout=timeout)


def refresh_repositories(repository=None, *, timeout=None):
    """Refresh the installed repositories=None"""
    _REPOSITORIES.refresh(repository=repository, timeout=timeout)


def get_bundles(repository, *, refresh=False, timeout=None):
    """Get all bundles in the repository

    Parameters
    ----------
    repository : str
        Name of the repository

    refresh : bool, optional
        Refresh the repository

        ..versionadded :: 1.1

    timeout : float, optional
        Timeout for json request

        ..versionadded :: 1.1

    Returns
    -------
    dict : A dict of key Bundle pairs
    """
    repository = _REPOSITORIES[repository]
    if refresh:
        repository.refresh(timeout)

    return repository.get_bundles()


def list_bundles(repository, *, refresh=False, timeout=None):
    """Get a list of all bundle names in the specified repository.

    Parameters
    ----------
    repository : str
        The name of the repository

    refresh : bool, optional
        Refresh the repository

        ..versionadded :: 1.1

    timeout : float, optional
        Timeout for json request

        ..versionadded :: 1.1

    Returns
    -------
    bundle : str
        The name of the bundle
    """
    return [
        key
        for key, bundle in get_bundles(
            repository, refresh=refresh, timeout=timeout
        ).items()
    ]


def list_collections(repository):
    """List the collections of the repository

    Parameters
    ----------
    repository : str or Bundle, optional
        The data repository

        - if str load a named bundle, format {repository}/{bundle}

    Returns
    -------
    list : a list of collections
    """
    (
        repository_name,
        bundle_name,
        bundle_version,
        bundle_tag,
    ) = _split_repo_bundle(repository)
    repository = get_repository(repository_name)
    collections = repository.get_bundle(bundle_name).collections
    return list(collections.keys()) if collections is not None else []


def list_repositories(*, refresh=False, timeout=None):
    """List the key of all installed repositories

    refresh : bool, optional
        Refresh all repositories

        ..versionadded :: 1.1

    timeout : float, optional
        Timeout for json request

        ..versionadded :: 1.1
    """
    if refresh:
        refresh_repositories(timeout=timeout)
    return [repo.name for repo in _REPOSITORIES]


# Install the default 'wildboar' repository
def _get_dataset_version():
    from pkg_resources import parse_version

    from wildboar import __version__

    v = parse_version(__version__)
    if v.is_prerelease:
        return "master"
    else:
        return "%d.%d" % (v.major, v.minor)


install_repository(
    "https://isaksamsten.github.io/wildboar-datasets/%s/repo.json"
    % _get_dataset_version(),
    refresh=False,
)

refresh_repositories(timeout=1)
