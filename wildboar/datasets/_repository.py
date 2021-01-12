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
from urllib.parse import urlparse

import numpy as np
import requests
from wildboar import __version__ as wildboar_version
from pkg_resources import parse_version
from scipy.io.arff import loadarff


class Repository:
    def __init__(self, url):
        self.url = url
        self.refresh()

    def get_bundle(self, key):
        repositories = self.get_bundles()
        return repositories.get(key)

    def get_bundles(self):
        return self.bundles

    def refresh(self):
        json = requests.get(self.url).json()
        self.wildboar_requires = json["wildboar_requires"]
        self.name = json["name"]
        self.version = json["version"]
        if parse_version(self.wildboar_requires) < parse_version(wildboar_version):
            raise ValueError(
                "repository requires wildboar (>=%s), got %s",
                self.wildboar_requires,
                wildboar_version,
            )
        self.url = json["url"]
        bundles = {}
        for bundle_json in json["bundles"]:
            key = bundle_json["key"]
            version = bundle_json["version"]
            name = bundle_json["name"]
            hash = bundle_json["hash"]
            class_index = bundle_json["class_index"]
            description = bundle_json["description"]
            download_url = self.url.format(key=key, version=version)
            bundles[bundle_json["key"]] = NpyBundle(
                key=key,
                version=version,
                name=name,
                download_url=download_url,
                description=description,
                hash=hash,
                class_index=class_index,
            )

        self.bundles = bundles


class RepositoryCollection:
    def __init__(self):
        self.repositories = []

    def __getitem__(self, item):
        return next(
            (repository for repository in self.repositories if repository.name == item),
            None,
        )

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

    download_url : str
        Local or remote path to the bundle

    hash : str
        SHA1 hash of the file pointed to by download_url

    class_index : int or array-like
        Index of the class label(s)
    """

    def __init__(
        self,
        key,
        version,
        name,
        download_url,
        *,
        description=None,
        hash=None,
        class_index=-1
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

        download_url : str
            Local or remote path to the bundle. file:// or http(s):// paths are supported.

        hash : str
            SHA1 hash of the file pointed to by download_url

        class_index : int or array-like
            Index of the class label(s)
        """
        self.key = key
        self.version = version
        self.name = name
        self.description = description
        self.download_url = download_url
        self.hash = hash
        self.class_index = class_index

    def list(self, cache_dir, *, create_cache_dir=True, progress=True, force=False):
        """List all datasets in this bundle

        Parameters
        ----------
        cache_dir : str
            Location of the cached download

        create_cache_dir : bool, optional
            Create cache directory if it does not exist.

        progress : bool, optional
            Write progress to standard error

        force : bool, optional
            Force re-download of cached bundle

        Returns
        -------
        dataset_names : list
            A sorted list of datasets in the bundle
        """
        with self._download_bundle(
            cache_dir=cache_dir,
            create_cache_dir=create_cache_dir,
            progress=progress,
            force=force,
        ) as archive:
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
        cache_dir,
        *,
        create_cache_dir=True,
        progress=True,
        dtype=None,
        force=False
    ):
        """Load a dataset from the bundle

        Parameters
        ----------
        name : str
            Name of the dataset

        cache_dir : str
            Location of the cached download

        create_cache_dir : bool, optional
            Create cache directory if it does not exist.

        progress : bool, optional
            Write progress to standard error

        dtype : object, optional
             Cast the data and label matrix to a specific type

        force : bool, optional
            Force re-download of cached bundle

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
        with self._download_bundle(
            cache_dir=cache_dir,
            create_cache_dir=create_cache_dir,
            progress=progress,
            force=force,
        ) as archive:
            datasets = []
            for dataset in map(_Dataset, archive.filelist):
                if dataset.filename == name and self._is_dataset(
                    dataset.path, dataset.ext
                ):
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

    def _download_bundle(
        self, cache_dir, *, create_cache_dir=True, progress=True, force=False
    ):
        """Download a bundle to the cache directory"""
        if not os.path.exists(cache_dir):
            if create_cache_dir:
                os.mkdir(cache_dir)
            else:
                raise ValueError(
                    "output directory does not exist (set create_cache_dir=True to create it)"
                )

        url_parse = urlparse(self.download_url)
        path = url_parse.path
        basename = os.path.basename(path)
        if basename == "":
            raise ValueError("expected .zip file got, %s" % basename)
        _, ext = os.path.splitext(basename)
        if ext != ".zip":
            raise ValueError("expected .zip file got, %s" % ext)

        filename = os.path.join(cache_dir, "%s-v%s" % (self.key, self.version))
        if os.path.exists(filename):
            if force:
                os.remove(filename)
            else:
                try:
                    z_file = zipfile.ZipFile(open(filename, "rb"))
                    self._check_integrity(filename)
                    return z_file
                except zipfile.BadZipFile:
                    os.remove(filename)
        if url_parse.scheme == "file":
            from shutil import copyfile

            copyfile(url_parse.path, filename)
        else:
            with open(filename, "wb") as f:
                response = requests.get(self.download_url, stream=True)
                if not response:
                    raise ValueError("file not found, %s" % self.download_url)
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
                                    self.download_url,
                                )
                            )
                            sys.stderr.flush()

        self._check_integrity(filename)
        return zipfile.ZipFile(open(filename, "rb"))

    def _check_integrity(self, filename):
        """Check the integrity of the downloaded or cached file"""
        if self.hash is not None:
            actual_hash = _sha1(filename)
            if self.hash != actual_hash:
                raise ValueError(
                    "integrity check failed, expected '%s', got '%s'"
                    % (self.hash, actual_hash)
                )


class ArffBundle(Bundle):
    """bundle of .arff-files"""

    def __init__(
        self,
        key,
        version,
        name,
        download_url,
        *,
        description=None,
        hash=None,
        class_index=-1,
        encoding="utf-8"
    ):
        super().__init__(
            key,
            version,
            name,
            download_url,
            hash=hash,
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
