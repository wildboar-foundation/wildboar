# Repository = namedtuple("Repository", "name description download_url hash extension ndim", defaults=[None, None, 1])
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
from scipy.io.arff import loadarff


class Repository(metaclass=ABCMeta):

    def __init__(self, name, download_url, *, description=None, hash=None, class_index=-1):
        self.name = name
        self.description = description
        self.download_url = download_url
        self.hash = hash
        self.class_index = class_index

    def list(self, *, cache_dir='wildboar_cache', create_cache_dir=True, progress=True):
        with self._download_repository(cache_dir=cache_dir, create_cache_dir=create_cache_dir,
                                       progress=progress) as archive:
            names = []
            for f in archive.filelist:
                path, ext = os.path.splitext(f.filename)
                if self._is_dataset(path, ext):
                    filename = os.path.basename(path)
                    filename = re.sub("_(TRAIN|TEST)", "", filename)
                    names.append(filename)

            return sorted(set(names))

    def load(self, name, *, dtype=None, cache_dir='wildboar_cache', create_cache_dir=True, progress=True):
        dtype = dtype or np.float64
        with self._download_repository(cache_dir=cache_dir, create_cache_dir=create_cache_dir,
                                       progress=progress) as archive:
            datasets = []
            for dataset in map(_Dataset, archive.filelist):
                if dataset.filename == name and self._is_dataset(dataset.path, dataset.ext):
                    datasets.append(dataset)

            if not datasets:
                raise ValueError("no dataset found (%s)" % name)
            train_parts = [self._load_array(archive, dataset.file) for dataset in datasets if dataset.part == 'train']
            test_parts = [self._load_array(archive, dataset.file) for dataset in datasets if dataset.part == 'test']

            data = np.vstack(train_parts)
            n_train_samples = data.shape[0]
            if test_parts:
                test = np.vstack(test_parts)
                data = np.vstack([data, test])

            y = data[:, self.class_index].astype(dtype)
            x = np.delete(data, self.class_index, axis=1).astype(dtype)
            return x, y, n_train_samples

    @abstractmethod
    def _is_dataset(self, path, ext):
        pass

    @abstractmethod
    def _load_array(self, archive, file):
        pass

    def _download_repository(self, cache_dir='wildboar_cache', create_cache_dir=True, progress=True):
        if not os.path.exists(cache_dir):
            if create_cache_dir:
                os.mkdir(cache_dir)
            else:
                raise ValueError("output directory does not exist (set create_cache_dir=True to create it)")

        url_parse = urlparse(self.download_url)
        path = url_parse.path
        basename = os.path.basename(path)
        if basename == "":
            raise ValueError("expected .zip file got, %s" % basename)
        _, ext = os.path.splitext(basename)
        if ext != ".zip":
            raise ValueError("expected .zip file got, %s" % ext)

        filename = os.path.join(cache_dir, basename)
        if os.path.exists(filename):
            try:
                z_file = zipfile.ZipFile(open(filename, 'rb'))
                self._check_integrity(filename)
                return z_file
            except zipfile.BadZipFile:
                os.remove(filename)
        if url_parse.scheme == "file":
            from shutil import copyfile
            copyfile(url_parse.path, filename)
        else:
            with open(filename, 'wb') as f:
                response = requests.get(self.download_url, stream=True)
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

    def _check_integrity(self, filename):
        if self.hash is not None:
            actual_hash = _sha1(filename)
            if self.hash != actual_hash:
                raise ValueError("integrity check failed, expected '%s', got '%s'" % (self.hash, actual_hash))


class ArffRepository(Repository):

    def __init__(self, name, download_url, *, description=None, hash=None, class_index=-1, encoding='utf-8'):
        super().__init__(name, download_url, hash=hash, description=description, class_index=class_index)
        self.encoding = encoding

    def _is_dataset(self, path, ext):
        return ext == ".arff"

    def _load_array(self, archive, file):
        with io.TextIOWrapper(archive.open(file), encoding=self.encoding) as io_wrapper:
            arff, _metadata = loadarff(io_wrapper)
            arr = np.array(arff.tolist())
            return arr


class NpyRepository(Repository):

    def _is_dataset(self, path, ext):
        return ext == '.npy'

    def _load_array(self, archive, file):
        return np.load(archive.open(file))


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


def _sha1(file, buf_size=65536):
    sha1 = hashlib.sha1()
    with open(file, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()
