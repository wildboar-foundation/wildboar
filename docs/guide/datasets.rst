========
Datasets
========
Wildboar is distributed with an advanced system for handling dataset repositories. A dataset repository can
be used to load benchmark datasets or to distribute or store datasets.

What is a repository?
=====================
I short, a repository is a collection of datasets. More specifically, a repository is a zip-file containing datasets
or dataset parts that can be downloaded, cached and loaded by wildboar.

How to use a repository?
========================
Repositories are either initialized directly or used together with the ``load_dataset`` function.

.. code-block:: python

    >>> from wildboar.datasets import load_dataset
    >>> x, y = load_dataset('GunPoint', repository='wildboar/ucr')
    # ... downloading repository to cache folder...
    >>> x.shape

.. note::

    Currently, wildboar bundles one collection of datasets, which can be downloaded from `timeseriesclassification.com <http://timeseriesclassification.com>`_
    distributed as two different repositories for completeness:

    wildboar/ucr
      Optimized for loading and download speed.

    timeseriesclassification/univariate
      Original dataset in .arff-format. Slower to load.

.. note::

    Repositories are cached locally in a folder controlled by the parameter ``cache_dir``. The default directory
    is a folder named 'wildboar_cache' relative to the current directory. To change the default cache-directory
    use ``DEFAULT_CACHE_DIR`` variable.

    .. code-block:: python

     >>> from wildboar.datasets import DEFAULT_CACHE_DIR
     >>> DEFAULT_CACHE_DIR = '/home/user/.cache/wildboar_dataset/'
     >>> load_dataset('Wafer', repository='wildboar/ucr')

    or

    .. code-block:: python

     >>> load_dataset("Wafer", repository="wildboar/ucr", cache_dir="/data/my_cache_drive")


How to create a repository?
===========================
To create a new (local or remote) repository one would create a collection of dataset-files, e.g.:

.. code-block:: python

    >>> x = np.arange(100 * 10).reshape(100, 10)
    >>> y = np.concatenate([np.ones(50), np.zeros(50)], axis=0).reshape(-1, 1)
    >>> np.save('folder_with_npy_files/example1.npy', np.hstack([x, y]), allow_pickle=False, fix_import=False)

and collect them in a zip-file (with a suitable utility, e.g., ``zip``):

.. code-block:: shell

    $ zip -r my_repo_npy.zip folder_with_npy_files/
    $ sha1sum my_repo_npy.zip
    464888e88e95d538ab6c134df0279a9776fd5dc7    my_repo_npy.zip

.. note::

    It is also possible to create or use `.arff`-files. However, the utilities for loading
    such files are not as performant as the binary representation of Numpy.

Once the datasets are created and collected a new repository can be constructed in Python.

.. code-block:: python

    >>> from wildboar.datasets import NpyRepository, load_dataset, install_repository
    >>> my_repo = NpyRepository(name="My Repository",
    ...                         download_url="file:///path/to/my_repo_npy.zip",
    ...                         hash="464888e88e95d538ab6c134df0279a9776fd5dc7",
    ...                         class_index=-1)
    >>> install_repository('my_repo', my_repo)
    >>> x, y = load_dataset('example1', repository='my_repo')


To avoid some of the boilerplate-code, ``load_dataset`` can directly infer and implicitly
construct a repository

.. code-block:: python

    >>> x, y = load_dataset('example1', repository="file:///path/to/my_repo_npy.zip")

.. note::

    When inferring the type of repository, the filename of the repository should end with
    the type of repository, e.g., ``_nyp`` or ``_arff``.

.. warning::

    Implicit repositories does not support repository integrity checks.

Repositories can also be uploaded and distributed over http(s):

.. code-block::

    >>> my_repo = NpyRepository(name="My Repository",
    ...                         download_url="https://example.com/my_repo_npy.zip",
    ...                         hash="464888e88e95d538ab6c134df0279a9776fd5dc7",
    ...                         class_index=-1)




