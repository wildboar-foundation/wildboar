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

import hashlib
import io
import os
import re
import sys
import zipfile
from abc import abstractmethod, ABCMeta

import numpy as np
import requests
from wildboar import __version__ as wildboar_version
from pkg_resources import parse_version
from scipy.io.arff import loadarff

DEFAULT_TAG = "default"


def _replace_placeholders(url, **kwargs):
    """Replace placeholder values of the format {key} with kwargs[key]

    Parameters
    ----------
    url : str
        The input string

    **kwargs : dict
        The key and values to replace

    Returns
    -------
    str : url with placeholder values replaced
    """
    for arg, value in kwargs.items():
        url = url.replace("{%s}" % arg, value)
    return url


def _check_integrity(bundle_file, hash_file):
    """Check the integrity of the downloaded or cached file

    Parameters
    ----------
    bundle_file : str, bytes or PathLike
        Path to the bundle file

    hash_file : str, bytes or PathLike
        Path to the hash file

    Returns
    -------
    bool : true if the hash of bundle file matches the contents of the hash file
    """
    with open(hash_file, "r") as f:
        hash = f.readline().strip()

    if hash is not None:
        actual_hash = _sha1(bundle_file)
        if hash != actual_hash:
            raise ValueError(
                "integrity check failed, expected '%s', got '%s'" % (hash, actual_hash)
            )
    return True


def _sha1_is_sane(hash_file):
    """Check the sanity of a hash file

    Parameters
    ----------
    hash_file : str, bytes or PathLike
        The path to the hash file

    Returns
    -------
    bool : Returns true if hash is 40 characters long; otherwise false.
    """
    with open(hash_file, "r") as f:
        return len(f.readline().strip()) == 40


def _load_archive(
    bundle_name,
    download_url,
    cache_dir,
    *,
    create_cache_dir=True,
    progress=True,
    force=False,
):
    """Load or download a bundle

    Parameters
    ----------
    bundle_name : str
        The name of the cached file

    download_url : str
        The download url to the bundle and hash file

    cache_dir : str
        The cache directory

    create_cache_dir : bool
        Create the cache directory if missing

    progress : bool
        Show progress information

    force : bool
        Remove any cached files and force re-download

    Returns
    -------
    archive : zipfile.ZipFile
        A zip-archive with datasets
    """
    if not os.path.exists(cache_dir):
        if create_cache_dir:
            os.makedirs(os.path.abspath(cache_dir), exist_ok=True)
        else:
            raise ValueError(
                "output directory does not exist (set create_cache_dir=True to create it)"
            )

    cached_hash = os.path.join(cache_dir, "%s.sha1" % bundle_name)
    cached_bundle = os.path.join(cache_dir, "%s.zip" % bundle_name)

    if force:
        if os.path.exists(cached_hash):
            os.remove(cached_hash)
        if os.path.exists(cached_bundle):
            os.remove(cached_bundle)

    if not os.path.exists(cached_hash) or not _sha1_is_sane(cached_hash):
        _download_hash_file(cached_hash, "%s.sha1" % download_url, bundle_name)

    if os.path.exists(cached_bundle) and _check_integrity(cached_bundle, cached_hash):
        try:
            return zipfile.ZipFile(open(cached_bundle, "rb"))
        except zipfile.BadZipFile:
            os.remove(cached_bundle)

    _download_bundle_file(cached_bundle, "%s.zip" % download_url, bundle_name, progress)
    _check_integrity(cached_bundle, cached_hash)
    return zipfile.ZipFile(open(cached_bundle, "rb"))


def _download_hash_file(cached_hash, hash_url, filename):
    """Download the

    Parameters
    ----------
    cached_hash : str, bytes or PathLike
        The path to the cached hash

    hash_url : str
        The download url

    filename : str
        The filename of the bundle
    """
    with open(cached_hash, "w") as f:
        response = requests.get(hash_url)
        if not response:
            f.close()
            os.remove(cached_hash)
            raise ValueError(
                "bundle (%s) not found (.sha1-file is missing). Try another version or tag."
                % filename
            )
        f.write(response.text)


def _download_bundle_file(cached_bundle, bundle_url, filename, progress):
    """Download the bundle

    Parameters
    ----------
    cached_bundle : str, bytes or PathLike
        The path to the cached bundle

    bundle_url : str
        The download url

    filename : str
        The filename of the bundle

    progress : bool
        Show progress bar
    """
    with open(cached_bundle, "wb") as f:
        response = requests.get(bundle_url, stream=True)
        if not response:
            f.close()
            os.remove(cached_bundle)
            raise ValueError(
                "bundle (%s) not found (.zip-file is missing). Try another version or tag."
                % filename
            )

        total_length = response.headers.get("content-length")
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
                    sys.stderr.write(
                        "\r[%s%s] %d/%d downloading %s"
                        % (
                            "=" * done,
                            " " * (50 - done),
                            length,
                            total_length,
                            filename,
                        )
                    )
                    sys.stderr.flush()


class Repository(metaclass=ABCMeta):
    """A repository is a collection of bundles"""

    @property
    @abstractmethod
    def name(self):
        """Name of the repository

        Returns
        -------
        str : the name of the repository
        """
        pass

    @property
    @abstractmethod
    def version(self):
        """The repository version

        Returns
        -------
        str : the version of the repository
        """
        pass

    @property
    @abstractmethod
    def download_url(self):
        """The url template for downloading bundles

        Returns
        -------
        str : the download url
        """
        pass

    @property
    @abstractmethod
    def wildboar_requires(self):
        """The minimum required wildboar version

        Returns
        -------
        str : the min version
        """
        pass

    @abstractmethod
    def get_bundles(self):
        """Get all bundles

        Returns
        -------
        dict : a dictionary of key and bundle
        """
        pass

    def get_bundle(self, key):
        """Get a bundle with the specified key

        Parameters
        ----------
        key : str
            Key of the bundle

        Returns
        -------
        bundle : Bundle, optional
            A bundle or None
        """
        bundle = self.get_bundles().get(key)
        if bundle is None:
            raise ValueError("bundle (%s) does not exist" % key)
        return bundle

    def load_dataset(
        self,
        bundle,
        dataset,
        *,
        cache_dir,
        version=None,
        tag=None,
        create_cache_dir=True,
        progress=True,
        dtype=None,
        force=False,
    ):
        bundle = self.get_bundle(bundle)
        version = version or bundle.version
        tag = tag or DEFAULT_TAG

        cache_dir = os.path.join(cache_dir, self.name)
        download_url = _replace_placeholders(
            self.download_url, bundle=bundle.key, version=version, tag=tag
        )

        with _load_archive(
            bundle_name=bundle.get_filename(version, tag),
            download_url=download_url,
            cache_dir=cache_dir,
            create_cache_dir=create_cache_dir,
            progress=progress,
            force=force,
        ) as archive:
            return bundle.load(dataset, archive, dtype=dtype)

    def list_datasets(
        self,
        bundle,
        *,
        cache_dir,
        version=None,
        tag=None,
        create_cache_dir=True,
        progress=True,
        force=False,
    ):
        bundle = self.get_bundle(bundle)
        version = version or bundle.version
        tag = tag or DEFAULT_TAG

        cache_dir = os.path.join(cache_dir, self.name)
        download_url = _replace_placeholders(
            self.download_url, bundle=bundle.key, version=version, tag=tag
        )
        with _load_archive(
            bundle_name=bundle.get_filename(version, tag),
            download_url=download_url,
            cache_dir=cache_dir,
            create_cache_dir=create_cache_dir,
            progress=progress,
            force=force,
        ) as archive:
            return bundle.list(archive)

    def clear_cache(self, cache_dir, keep_last_version=True):
        cache_dir = os.path.join(cache_dir, self.name)
        if not os.path.exists(cache_dir) or not os.path.isdir(cache_dir):
            return

        keep = []
        if keep_last_version:
            keep = [
                bundle.get_filename(tag=DEFAULT_TAG)
                for _, bundle in self.get_bundles().items()
            ]

        for filename in os.listdir(cache_dir):
            basename, ext = os.path.splitext(filename)
            full_path = os.path.join(cache_dir, filename)
            if (
                os.path.isfile(full_path)
                and ext in [".zip", ".sha1"]
                and not basename in keep
            ):
                os.remove(full_path)

    def refresh(self):
        """Refresh the repository"""
        pass


def _validate_url(url):
    if "{bundle}" in url and "{version}" in url and "{tag}" in url:
        return url
    else:
        raise ValueError(
            "repository url is invalid, got %s ({bundle}, {version} and {tag} are required)"
            % url
        )


def _validate_repository_name(str):
    if re.match("[a-zA-Z]+", str):
        return str
    else:
        raise ValueError("repository name (%s) is not valid" % str)


def _validate_bundle_key(str):
    if re.match("[a-zA-Z0-9\-]+", str):
        return str
    else:
        raise ValueError("bundle key (%s) is not valid" % str)


def _validate_version(str):
    if re.match("(^(?:\d+\.)?(?:\d+\.)?(?:\*|\d+)$)", str):
        return str
    else:
        raise ValueError("version (%s) is not valid" % str)


class JSONRepository(Repository):
    def __init__(self, url):
        self.repo_url = url
        self.__refresh()

    @property
    def wildboar_requires(self):
        return self._wildboar_requires

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def download_url(self):
        return self._bundle_url

    def get_bundles(self):
        return self._bundles

    def refresh(self):
        self.__refresh()

    def __refresh(self):
        json = requests.get(self.repo_url).json()
        self._wildboar_requires = json["wildboar_requires"]
        self._name = _validate_repository_name(json["name"])
        self._version = _validate_version(json["version"])
        if parse_version(self.wildboar_requires) > parse_version(wildboar_version):
            raise ValueError(
                "repository requires wildboar (>=%s), got %s",
                self.wildboar_requires,
                wildboar_version,
            )
        self._bundle_url = _validate_url(json["bundle_url"])
        bundles = {}
        for bundle_json in json["bundles"]:
            key = _validate_bundle_key(bundle_json["key"])
            version = _validate_version(bundle_json["version"])
            name = bundle_json["name"]
            class_index = bundle_json["class_index"]
            description = bundle_json["description"]
            bundles[bundle_json["key"]] = NpyBundle(
                key=key,
                version=version,
                name=name,
                description=description,
                class_index=class_index,
            )

        self._bundles = bundles


class RepositoryCollection:
    def __init__(self):
        self.repositories = []

    def __getitem__(self, item):
        return next(
            (repository for repository in self.repositories if repository.name == item),
            None,
        )

    def __contains__(self, item):
        return any(r for r in self.repositories if r.name == item)

    def __iter__(self):
        return iter(self.repositories)

    def append(self, item):
        if self[item]:
            raise ValueError("cannot overwrite repository, %s" % item.name)
        self.repositories.append(item)


class Bundle(metaclass=ABCMeta):
    """Base class for handling dataset bundles

    Attributes
    ----------

    name : str
        Human-readable name of the bundle

    description : str
        Description of the bundle

    class_index : int or array-like
        Index of the class label(s)
    """

    def __init__(
        self,
        *,
        key,
        version,
        name,
        description=None,
        class_index=-1,
    ):
        """Construct a bundle

        Parameters
        ----------
        key : str
            A unique key of the bundle

        version : str
            The version of the bundle

        name : str
            Human-readable name of the bundle

        description : str
            Description of the bundle

        class_index : int or array-like
            Index of the class label(s)
        """
        self.key = key
        self.version = version
        self.name = name
        self.description = description
        self.class_index = class_index

    def get_filename(self, version=None, tag=None, ext=None):
        filename = "%s-v%s" % (self.key, version or self.version)
        if tag:
            filename += ":%s" % tag
        if ext:
            filename += ext
        return filename

    def list(self, archive):
        """List all datasets in this bundle

        Parameters
        ----------
        archive : ZipFile
            The bundle file

        Returns
        -------
        dataset_names : list
            A sorted list of datasets in the bundle
        """
        names = []
        for f in archive.filelist:
            path, ext = os.path.splitext(f.filename)
            if self._is_dataset(path, ext):
                filename = os.path.basename(path)
                filename = re.sub("_(TRAIN|TEST)", "", filename)
                names.append(filename)

        return sorted(set(names))

    def load(
        self,
        name,
        archive,
        *,
        dtype=None,
    ):
        """Load a dataset from the bundle

        Parameters
        ----------
        name : str
            Name of the dataset

        archive : ZipFile
            The zip-file bundle

        dtype : object, optional
            Cast the data and label matrix to a specific type

        Returns
        -------
        x : ndarray
            Data samples

        y : ndarray
            Data labels

        n_training_samples : int
            Number of samples that are for training. The value is <= x.shape[0]
        """
        dtype = dtype or np.float64
        datasets = []
        for dataset in map(_Dataset, archive.filelist):
            if dataset.filename == name and self._is_dataset(dataset.path, dataset.ext):
                datasets.append(dataset)

        if not datasets:
            raise ValueError("no dataset found (%s)" % name)
        train_parts = [
            self._load_array(archive, dataset.file)
            for dataset in datasets
            if dataset.part == "train"
        ]
        test_parts = [
            self._load_array(archive, dataset.file)
            for dataset in datasets
            if dataset.part == "test"
        ]

        data = np.vstack(train_parts)
        n_train_samples = data.shape[0]
        if test_parts:
            test = np.vstack(test_parts)
            data = np.vstack([data, test])

        y = data[:, self.class_index].astype(dtype)
        x = np.delete(data, self.class_index, axis=1).astype(dtype)
        return x, y, n_train_samples

    @abstractmethod
    def _is_dataset(self, file_name, ext):
        """Overridden by subclasses

        Check if a path and extension is to be considered a dataset. The check should be simple and only consider
        the filename and or extension of the file. Validation of the file should be deferred to `_load_array`

        Parameters
        ----------
        file_name : str
            Name of the dataset file

        ext : str
            Extension of the dataset file

        Returns
        -------
        bool
            True if it is a dataset
        """
        pass

    @abstractmethod
    def _load_array(self, archive, file):
        """Overridden by subclasses

        Load the file inside the archive and convert to a numpy array

        Parameters
        ----------
        archive : ZipFile
            The zip-archive in which the file reside

        file : str
            Path to the file inside the zip-archive

        Returns
        -------
        ndarray
            The dataset converted to a ndarray
        """
        pass


class ArffBundle(Bundle):
    """bundle of .arff-files"""

    def __init__(
        self,
        *,
        key,
        version,
        name,
        description=None,
        class_index=-1,
        encoding="utf-8",
    ):
        super().__init__(
            key=key,
            version=version,
            name=name,
            description=description,
            class_index=class_index,
        )
        self.encoding = encoding

    def _is_dataset(self, file_name, ext):
        return ext == ".arff"

    def _load_array(self, archive, file):
        with io.TextIOWrapper(archive.open(file), encoding=self.encoding) as io_wrapper:
            arff, _metadata = loadarff(io_wrapper)
            arr = np.array(arff.tolist())
            return arr


class NpyBundle(Bundle):
    """bundle of numpy binary files"""

    def _is_dataset(self, file_name, ext):
        return ext == ".npy"

    def _load_array(self, archive, file):
        return np.load(archive.open(file))


class _Dataset:
    def __init__(self, zip_info):
        self.file = zip_info
        self.path, self.ext = os.path.splitext(zip_info.filename)
        self.filename = os.path.basename(self.path)
        if "_TRAIN" in self.filename:
            self.part = "train"
            self.filename = self.filename.replace("_TRAIN", "")
        elif "_TEST" in self.filename:
            self.part = "test"
            self.filename = self.filename.replace("_TEST", "")
        else:
            self.part = "train"


def _sha1(file, buf_size=65536):
    sha1 = hashlib.sha1()
    with open(file, "rb") as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()
