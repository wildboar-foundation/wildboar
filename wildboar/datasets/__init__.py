import os
import re
from urllib.parse import urlparse

import numpy as np

from . import _resources
from ._repository import ArffRepository, NpyRepository, Repository

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

_REPOSITORIES = {
    'timeseriesclassification/univariate': ArffRepository(
        name="UCR Time series repository, univariate",
        description="A collection of 128 univariate time series. Downloaded from UCR Time series repository",
        download_url="http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip",
        hash="db696c772f0a0c2679fc41fed7b2bbe6a67af251",
        class_index=-1,
        encoding='utf-8'
    ),
    'wildboar/ucr': NpyRepository(
        name="UCR Time series repository, univariate (Numpy optimized)",
        description="A collection of 128 univariate time series. Downloaded from UCR Time series repository",
        download_url="https://github.com/isaksamsten/wildboar/releases/download/dataset-v1.0/ucr_2018_npy.zip",
        hash="11bacbb31ccfe928be38740107dab38218ac50fa",
        class_index=-1
    )
}

_REPOSITORY_INFERENCE_TYPES = {'npy', 'arff'}

__all__ = [
    "Repository",
    "ArffRepository",
    "NpyRepository",
    "get_repository",
    "install_repository",
    "load_dataset",
    "load_all_datasets",
    "load_two_lead_ecg",
    "load_synthetic_control",
    "load_gun_point",
]

_BUNDLED_DATASETS = {
    'synthetic_control': 'synthetic_control',
    'two_lead_ecg': 'TwoLeadECG',
    'gun_point': "Gun_Point",
    'shapelet_sim': 'ShapeletSim',
    'insect_wing_beat_sound': 'InsectWingbeatSound',
    'arrow_head': 'ArrowHead'
}


def load_synthetic_control(**kvargs):
    """Load the Synthetic_Control dataset

    See Also
    --------
    load_dataset : load a named dataset
    """
    return load_dataset("synthetic_control", **kvargs)


def load_two_lead_ecg(**kwargs):
    """Load the TwoLeadECG dataset

    See Also
    --------
    load_dataset : load a named dataset
    """
    return load_dataset('two_lead_ecg', **kwargs)


def load_gun_point(**kwargs):
    """Load the GunPoint dataset

    See Also
    --------
    load_dataset : load a named dataset
    """
    return load_dataset('gun_point', **kwargs)


def load_all_datasets(*, repository=None, cache_dir='wildboar_cache', create_cache_dir=True, progress=True, **kwargs):
    """Load all datasets as a generator

    Parameters
    ----------
    repository : str
        A string with the repository.

    progress : bool, optional
        If progress indicator is shown while downloading the bundle.

    cache_dir : str, optional
        The cache directory for downloaded dataset bundles.

    create_cache_dir : bool, optional
        Create the cache directory if it does not exist.

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
    >>> for dataset, (x, y) in load_all_datasets(repository='wildboar/ucr'):
    >>>     print(dataset, x.shape, y.shape)

    """

    for dataset in list_datasets(repository=repository, cache_dir=cache_dir, create_cache_dir=create_cache_dir,
                                 progress=progress):
        yield dataset, load_dataset(dataset, repository=repository, **kwargs)


def load_dataset(name, *, repository=None, dtype=None, contiguous=True, merge_train_test=True, **kwargs):
    """Load a dataset from a repository

    Parameters
    ----------
    name : str
        The name of the dataset to load.

    repository : str or Repository, optional
        The data repository

        - if `None` load one of the bundled data sets
        - if str load a named repository
        - if str http(s) or file url, load it as a bundle

    dtype : dtype, optional, default=np.float64
        The data type of the returned data

    contiguous : bool, optional
        Ensure that the returned dataset is memory contiguous.

    merge_train_test : bool, optional
        Merge the existing training and testing partitions.

    **kwargs : dict
        Additional arguments


    Other Parameters
    ----------------
    progress: bool, optional
        Show a progress bar while downloading a bundle.

    cache_dir: str, optional
        The directory where downloaded files are cached (default='wildboar_cache')

    create_cache_dir: bool, optional
        Create cache directory if missing (default=True)

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
    - To support other data formats create subclasses ``Repository``.
    - If an url is given as repository, the type of bundle is inferred from the file name.

    Examples
    --------

    Load one of the bundled datasets

    >>> x, y = load_dataset("synthetic_control")

    or if original training and testing splits are to be preserved

    >>> x_train, x_test, y_train, y_test = load_dataset("synthetic_control", merge_train_test=False)

    one can specify a different repository

    >>> x, y = load_dataset('Adiac', repository='timeseriesclassification/univariate')

    and with training and testing parts

    >>> x_train, x_test, y_train, y_test = load_dataset("Wafer", repository='wildboar/ucr', merge_train_test=False)

    or a selfhosted repository inferring the repository type from the bundle

    >>> x, y = load_dataset('my_data', repository="https://example.org/my_repository_arff.zip")

    or a selfhosted repository

    >>> x, y = load_dataset("my_data", repository=NpyRepository("my_repo",download_url="https://example.org/my_repository.zip"))

    """
    dtype = dtype or np.float64
    ret_val = []
    if repository is None:
        x, y, n_train_samples = _load_bundled_dataset(name, dtype)
    else:
        repository = get_repository(repository)
        x, y, n_train_samples = repository.load(name, dtype=dtype, **kwargs)

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


def list_datasets(repository=None, cache_dir='wildboar_cache', create_cache_dir=True, progress=True):
    """List the datasets in the repository

    Parameters
    ----------
    repository : str or Repository, optional
        The data repository

        - if `None` load one of the bundled data sets
        - if str load a named repository
        - if str http(s) or file url, load it as a bundle

    Other Parameters
    ----------------
    progress: bool, optional
        Show a progress bar while downloading a bundle.

    cache_dir: str, optional
        The directory where downloaded files are cached (default='wildboar_cache')

    create_cache_dir: bool, optional
        Create cache directory if missing (default=True)

    Returns
    -------
        dataset : set
            A set of dataset names
    """
    if repository is None:
        return _BUNDLED_DATASETS.keys()
    else:
        repository = get_repository(repository)
        return repository.list(cache_dir=cache_dir, create_cache_dir=create_cache_dir, progress=progress)


def get_repository(repository):
    """Get a repository from a str

    Parameters
    ----------
    repository : str
        Find a named repository or construct a new repository from the url

    Returns
    -------
    repository : Repository
        A repository

    Examples
    --------

    Load a named repositoru

    >>> repository = get_repository("wildboar/ucr")
    >>> x, y, n_train_samples = repository.load("Wafer", dtype=np.float)

    or from a url

    >>> repository = get_repository("https://example.org/my_repository_arff.zip")
    >>> x, y, n_train_samples = repository.load("my_data")
    """
    if repository in _REPOSITORIES.keys():
        return _REPOSITORIES[repository]
    elif isinstance(repository, Repository):
        return repository
    elif re.match('(http|https|file)://', repository):
        url = urlparse(repository)
        filename = os.path.basename(url.path)
        name, ext = os.path.splitext(filename)
        if ext != ".zip":
            raise ValueError("only .zip repositories are supported")
        repository_inference = re.match(r".*?_([a-zA-Z]+)", name)
        if repository_inference:
            repository_type = repository_inference.group(1)
            if repository_type in _REPOSITORY_INFERENCE_TYPES:
                return _new_repository(name, "Temporary repository", download_url=repository,
                                       extension=".%s" % repository_type, class_index=-1, hash=None)
            else:
                raise ValueError("repository (%s) is not supported" % repository_type)
        else:
            raise ValueError("unable to infer the repository type")

    else:
        raise ValueError("repository (%s) is not supported" % repository)


def install_repository(name, *, repository=None, download_url=None, description=None, hash=None, class_index=-1,
                       extension=None):
    """ Install a named repository

    Parameters
    ----------
    name : str
        The name of the repository.

    repository : Repository, optional
        Install repository if it exists

    download_url : str, optional
        If repository is None, create a new Repository from this bundle url.

    description : str, optional
        If repository is None, create a new Repository with this description

    hash : str, optional
        If repository is None, create a new Repository with this hash

    class_index : int, optional
        If repository is None, create a new Repository with this class index

    extension : str, optional
        If repository is None, create a new Repository with this type
    """
    if repository:
        _REPOSITORIES[name] = repository
    elif download_url is not None and extension is not None:
        _REPOSITORIES[name] = _new_repository(name, description, download_url, hash, class_index, extension)
    else:
        raise ValueError("not a valid repository")


def _new_repository(name, description, download_url, hash, class_index, extension):
    if extension == '.arff':
        return ArffRepository(name, download_url, description=description, hash=hash, class_index=class_index)
    elif extension == '.npy':
        return NpyRepository(name, download_url, description=description, hash=hash, class_index=class_index)
    else:
        raise ValueError("extension (%s) is not supported" % extension)


def _load_bundled_dataset(name, dtype):
    if name not in _BUNDLED_DATASETS:
        raise ValueError("dataset (%s) does not exist" % name)
    name = _BUNDLED_DATASETS[name]
    train_file = pkg_resources.open_text(_resources, "%s_TRAIN.txt" % name)
    test_file = pkg_resources.open_text(_resources, "%s_TEST.txt" % name)
    train = np.loadtxt(train_file, delimiter=",")
    test = np.loadtxt(test_file, delimiter=",")
    n_train_samples = train.shape[0]
    train = np.vstack([train, test])
    y = train[:, 0].astype(dtype)
    x = train[:, 1:].astype(dtype)
    return x, y, n_train_samples
