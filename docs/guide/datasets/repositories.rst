.. currentmodule:: wildboar
.. _guide-datasets-repositories:

############
Repositories
############

We can either initialize repositories directly or use them together with the
:func:`~datasets.load_dataset`-function:

.. code-block:: python

   from wildboar.datasets import load_dataset
   x, y = load_dataset('GunPoint', repository='wildboar/ucr')

Installed repositories and dataset bundles can be listed using the function
:func:`~datasets.list_repositories` and :func:`~datasets.list_bundles`
respectively.

.. code-block:: python

   >>> from wildboar.datasets import list_repositories, list_bundles, list_datasets
   >>> list_repositories()
   ['wildboar']
   >>> list_bundles("wildboar")
   ['ucr', 'ucr-tiny', ... (and more)]
   >>> list_datasets("wildboar/ucr-tiny")
   ['Beef', 'Coffee', 'GunPoint', 'SyntheticControl', 'TwoLeadECG']

**********************
Repository definitions
**********************

A wildboar repository string is composed of 2 required and one optional
components written as:

.. versionchanged:: 1.2

   The ``{version}`` specifier has been removed. The version is determined by the repository.

::

   {repository}/{bundle}[:{tag}]
   └─────┬────┘ └───┬──┘└───┬──┘
         │          │       └── (optional) The tag as defined below.
         │          └── (required) The bundle as listed by list_bundles().
         └── (required) The repository as listed by list_repositories().

Each part of the repository has the following requirements:

``{repository}``
   The repository identifier, as listed by :func:`datasets.list_repositories`.
   The identifier is composed of letters, i.e., matching the regular
   expression, `\w+`.

``{bundle}``
   The bundle identifier, as listed by :func:`datasets.list_bundles`. The
   identifier is composed of alphanumeric characters and ``-``, matching the
   regular expression ``[a-zA-Z0-9\-]+``.

``{tag}``
   The bundle tag (defaults to ``default``). The bundle tag is composed of
   letters and ``-``, matching the regular expression ``[a-zA-Z-]+``.

To exemplify, these are valid repository declarations:

``wildboar/ucr``
   The ``ucr`` bundle from the ``wildboar`` repository using the default tag.

``wildboar/ucr-tiny``
   The ``ucr-tiny`` bundle from the ``wildboar`` repository using the default tag.

``wildboar/outlier:hard``
   The ``outlier`` bundle, from the ``wildboar`` repository using the tag ``hard``.

***********************
Installing repositories
***********************

A repository implements the interface of the class
:class:`~datasets.Repository`.

.. note::

   The default repository (``wildboar``) is loaded by the class
   :class:`~datasets.JSONRepository`, which can load datasets specified by a
   JSON endpoint.

Repositories are installed using the function
:func:`~datasets.install_repository` which takes either a URL to a JSON-file or
an instance of (or a class implementing the interface of)
:class:`~datasets.Repository`.

.. code-block:: python

   >>> from wildboar.datasets import install_repository
   >>> install_repository("https://www.example.org/repo.json")
   >>> list_repositories("example")
   >>> load_dataset("example", repository="example/example")

Repositories can be refreshed using :func:`datasets.refresh_repositories()`,
which accepts a repository name to refresh a specific repository or :python:`None`
(default) to refresh all repositories. Additionally, we can specify an optional
refresh timeout (in seconds), and an optional cache location.

.. versionchanged:: 1.1

   Wildboar caches the repository definition locally to allow cached datasets to be
   used while offline.

Local cache
===========

Wildboar downloads, on-demand, datasets the first time we request a bundle and
caches it to disk in a directory determined by the operating system.
Wildboar caches datasets and repositories in the following directories:

**Windows**
   ``%LOCALAPPDATA%\cache\wildboar``

**GNU/Linux**
   ``$XDG_CACHE_HOME/wildboar``. If ``$XDG_CACHE_HOME`` is unset, we default to `.cache`.

**macOS**
   ``~/LibraryCaches/wildboar``.

**Fallback**
   ``~/.cache/wildboar``

The user can change the cache directory, either globally (for as long as the
current Python session lasts) with :func:`datasets.set_cache_dir` or locally
(for a specific operation) with then ``cache_dir``-parameter:

.. code-block:: python

   >>> from wildboar.datasets import set_cache_dir
   >>> set_cache_dir("/path/to/wildboar-cache/") # Set the global cache
   >>> load_dataset("GunPoint", cache_dir="/path/to/another/wildboar-cache/") # Another, local, cache here

If called without arguments, :func:`~datasets.set_cache_dir` resets the cache
to the default location based on the operating system.

*****************
JSON repositories
*****************

By default, repositories installed with {func}`datasets.install_repository`
should point to a JSON-file, which describes the available datasets and the
location where Wildboar can download them. The repository declaration is a
JSON-file:

.. code-block:: json

   {
      "name": "example", // required
      "version": "1.0",  // required
      "wildboar_requires": "1.1", // required, the minimum required wildboar version
      "bundle_url": "https://example.org/download/{key}/{tag}-v{version}", // required, the data endpoint
      "bundles": [ // required
         {
         "key": "example", // required, unique key of the bundle
         "version": "1.0", // required, the default version of dataset
         "tag": "default"  // optional, the default tag
         "name": "UCR Time series repository", // required
         "description": "Example dataset", // optional
         "arrays": ["x", "y"] // optional
         "collections": {"key": ["example1", "example"]} // optional
         },
      ]
   }

- The attributes ``{key}``, ``{version}`` and ``{tag}`` in the ``bundle_url`` are
  replaced with the bundle-key, bundle-version and bundle tag from the
  repository string. All attributes are required in the URL.

- The ``arrays`` attribute is optional. However, if it is not present, the dataset
  is assumed to be a single Numpy array, where the last column contains the
  class label or a Numpy-dict with both ``x``, and ``y`` keys.

  - if any other value except ``x`` and/or ``y`` is present in the
    ``arrays``-list, it will be loaded as an `extras`-dictionary and only
    returned if requested by the user.
  - if `y` is not present in arrays :func:`~datasets.load_dataset` return
    :python:`None` for `y`

- The ``bundles/version`` attribute, is the version of the bundle.

- The ``bundles/tag`` attribute is the default tag of the bundle which is used
  unless the user specifies an alternative bundle. If not specified, the tag is
  `default`.

- The ``bundles/collections`` attribute is a dictionary of named collections of
  datasets which can be specified when using
  :python:`load_datasets(..., collection="key")`

The ``bundle_url`` points to a remote location that for each bundle ``key``,
contains two files with extensions ``.zip`` and ``.sha`` respectively. In the
example, ``bundle_url`` should contain the two files
``example/default-v1.0.zip`` and ``example/default-v1.0.sha`` The ``.sha``-file
should contain the ``sha1`` hash of the ``.zip``-file to ensure the integrity
of the downloaded file. The ``zip``-file should contain the datasets.

By default, wildboar supports dataset bundles formatted as ``zip``-files
containing `npy` or `npz`-files, as created by :func:`numpy.save` and
:func:`numpy.savez`. The datasets in the ``zip``-file must be named according
to the regular expression ``{dataset_name}(_TRAIN|_TEST)?.(npy|npz)``. That is,
the dataset name (as specified when using `load_dataset`) and optionally
`_TRAIN` or ``_TEST`` followed by the extension ``npy`` or ``npz``. If there
are multiple datasets with the same name but different training or testing
tags, they will be merged. As such, if both ``_TRAIN`` and ``_TEST`` files are
present for the same name, ``load_dataset`` can return these train and test
samples separately by setting :python:`merge_train_test=False`. For example,
the ``ucr``-bundle provides the default train/test splits from the UCR time
series repository.

.. code-block:: python

   from wildboar.datasets import load_dataset
   x_train, x_test, y_train, y_test = load_dataset(
      'GunPoint', repository='wildboar/ucr', merge_train_test=False
   )
