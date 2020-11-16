import hashlib
import re

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

import zipfile
import requests
from urllib.parse import urlparse
import os
import io
import sys
import numpy as np
from scipy.io.arff import loadarff

_DATASET_SOURCES = {
    'ucr_univariate':
        ("http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip",
         "db696c772f0a0c2679fc41fed7b2bbe6a67af251"),
    'ucr_multivariate':
        ("http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip",
         "04527f9c3ab66a862c24db43dc234cac9a679830")
}

from . import _resources

__all__ = ["load_dataset", "load_all_datasets", "load_two_lead_ecg", "load_synthetic_control", "load_gun_point"]

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


def load_all_datasets(repository=None, **kwargs):
    """Load all datasets as a generator

    Parameters
    ----------
    repository : {'ucr'}, str
        A string with the repository

    dtype : dtype, optional, default=np.float64
        The dtype of the returned data

    contiguous : bool, optional
        Ensure that the returned dataset is memory contiguous

    merge_train_test : bool, optional
        Merge the existing training and testing partitions

    Yields
    ------
    x : array-like
        Data samples

    y : array-like
        Data labels

    Examples
    --------

    >>> from wildboar.datasets import load_all_datasets
    >>> for dataset, (x, y) in load_all_datasets(dtype=np.float64):
    >>>     pass # use x and y

    """
    for dataset in list_datasets(repository=repository, **kwargs):
        yield dataset, load_dataset(dataset, repository=repository, **kwargs)


def load_dataset(name, repository=None, dtype=None, contiguous=True, merge_train_test=True, **kwargs):
    """
    Load a dataset

    Parameters
    ----------
    name : str
        The name of the dataset to load.

    repository : {None, 'ucr'} or str
        The data repository

        - if `None` load one of the bundled data sets
        - if 'ucr', 'ucr_univariate' load datasets from the UCR repository
        - if str an url is expected to download a zip-file with '.arff'-files.
          Valid protocols: http://, https:// and file://.

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
    url: str, optional
        Url to download the dataset bundle from.

    sha1: str, optional
        The sha1 of the dataset bundle. If None, the integrity of the repository is not checked.

    progress: bool, optional
        Show a progress bar while downloading a bundle.

    cache_dir: str, optional
        The directory where downloaded files are cached (default='wildboar_cache')

    create_cache_dir: bool, optional
        Create cache directory if missing (default=True)

    encoding: str, optional
        The text encoding of the dataset files (default='utf-8')

    class_index: int, optiona
        The column of the class (default=-1)

    extension: str
        The extension of data files (default='.arff)


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
    A dataset bundle is defined as zip-file. Files matching ``extension`` will be considered as dataset parts.
    Parts sharing the same name will be merged (two files with the same name in different folders share
    name). Filenames (without extension) with the suffix '_TRAIN' or '_TEST' are considered as training
    and testing parts and are used togheter with the attribute ``merge_train_test=False``. Parts without any suffix
    are considered as training parts.

    Warnings
    --------
    Currently only '.arff'-files are supported.

    Examples
    --------

    >>> x, y = load_dataset("synthetic_control")

    or if original training and testing splits are to be preserved

    >>> x_train, x_test, y_train, y_test = load_dataset("synthetic_control", merge_train_test=False)

    one can specify a different repository

    >>> x, y = load_dataset('Adiac', repository='ucr')

    and with training and testing parts

    >>> x_train, x_test, y_train, y_test = load_dataset("Wafer", repository='ucr', merge_train_test=False)

    or a selfhosted repository

    >>> x, y = load_dataset('my_data', repository="https://example.org/my_repository.zip")
    """
    dtype = dtype or np.float64
    ret_val = []
    if repository is None:
        x, y, n_train_samples = _load_bundled_dataset(name, dtype)
    else:
        url, sha1 = _get_repository_url(repository)
        if sha1 is None:
            sha1 = kwargs.pop('sha1', None)

        x, y, n_train_samples = _load_univariate_repository(name, url, dtype, sha1, **kwargs)

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


def list_datasets(repository=None, **kwargs):
    if repository is None:
        return _BUNDLED_DATASETS.keys()
    else:
        url, sha1 = _get_repository_url(repository)
        with _download_repository(url, sha1, **kwargs) as archive:
            names = []
            for f in archive.filelist:
                path, ext = os.path.splitext(f.filename)
                if ext == '.arff':
                    filename = os.path.basename(path)
                    filename = re.sub("_(TRAIN|TEST)", "", filename)
                    names.append(filename)

            return set(names)


def _get_repository_url(repository):
    if repository in ['ucr', 'ucr_univariate']:
        url, sha1 = _DATASET_SOURCES['ucr_univariate']
    elif re.match('(http|https|file)://', repository):
        url = repository
        sha1 = None
    else:
        raise ValueError("repository (%s) is not supported" % repository)
    return url, sha1


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


def _download_repository(url, sha1, cache_dir='wildboar_cache', create_cache_dir=True, progress=True):
    if not os.path.exists(cache_dir):
        if create_cache_dir:
            os.mkdir(cache_dir)
        else:
            raise ValueError("output directory does not exist (set `create_out_dir=True` to create it)")

    path = urlparse(url).path
    basename = os.path.basename(path)
    if basename == "":
        raise ValueError("expected .zip file got, %s" % basename)
    _, ext = os.path.splitext(basename)
    if ext != ".zip":
        raise ValueError("expected .zip file got, %s" % ext)

    filename = os.path.join(cache_dir, basename)
    if os.path.exists(filename):
        try:
            _check_integrity(filename, sha1)
            return zipfile.ZipFile(open(filename, 'rb'))
        except zipfile.BadZipFile:
            os.remove(filename)

    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')
        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            length = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                length += len(data)
                f.write(data)
                done = int(50 * length / total_length)
                if length % 10 == 0 and progress:
                    sys.stderr.write("\r[%s%s] %d/%d downloading %s" %
                                     ('=' * done, ' ' * (50 - done), length, total_length, basename))
                    sys.stderr.flush()

    return zipfile.ZipFile(open(filename, 'rb'))


def _check_integrity(filename, sha1):
    if sha1 is not None:
        actual_sha1 = _sha1(filename)
        if sha1 != actual_sha1:
            raise ValueError("integrity check failed, expected '%s', got '%s'" % (sha1, actual_sha1))


class _Dataset:

    def __init__(self, zip_info):
        self.file = zip_info
        self.path, self.ext = os.path.splitext(zip_info.filename)
        self.filename = os.path.basename(self.path)
        if "_TRAIN" in self.filename:
            self.part = 'train'
            self.filename = self.filename.replace("_TRAIN", "")
        elif "_TEST" in self.filename:
            self.part = 'test'
            self.filename = self.filename.replace("_TEST", "")
        else:
            self.part = 'train'

    def load_array(self, archive, dtype=np.float64, encoding=None):
        if self.ext == '.arff':
            with io.TextIOWrapper(archive.open(self.file), encoding=encoding) as io_wrapper:
                arff, _metadata = loadarff(io_wrapper)
                arr = np.array(arff.tolist())
                return arr.astype(dtype)
        else:
            raise ValueError("ext (%s) not supported" % self.ext)


def _load_univariate_repository(name, url, dtype, sha1, class_index=-1, extension='.arff', encoding='utf-8', **kwargs):
    with _download_repository(url, sha1, **kwargs) as archive:
        datasets = []
        for dataset in map(_Dataset, archive.filelist):
            if dataset.filename == name and dataset.ext in extension:
                datasets.append(dataset)

        if not datasets:
            raise ValueError("no dataset found (%s)" % name)
        train_parts = [dataset.load_array(archive, dtype, encoding) for dataset in datasets if dataset.part == 'train']
        test_parts = [dataset.load_array(archive, dtype, encoding) for dataset in datasets if dataset.part == 'test']

        data = np.vstack(train_parts)
        n_train_samples = data.shape[0]
        if test_parts:
            test = np.vstack(test_parts)
            data = np.vstack([data, test])

        y = data[:, class_index].astype(dtype)
        x = np.delete(data, class_index, axis=1).astype(dtype)
        return x, y, n_train_samples


def _sha1(file, buf_size=65536):
    sha1 = hashlib.sha1()
    with open(file, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()
