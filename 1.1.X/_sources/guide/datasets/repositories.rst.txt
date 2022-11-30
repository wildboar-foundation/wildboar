============
Repositories
============

Repositories are either initialized directly or used together with the ``load_dataset`` function.

.. code-block:: python

    from wildboar.datasets import load_dataset
    x, y = load_dataset('GunPoint', repository='wildboar/ucr')
    # ... downloading repository to cache folder...

Installed repositories and dataset bundles can be listed using the function
``list_repositories`` and ``list_bundles`` respectively.

.. code-block:: python

 from wildboar.datasets import list_repositories, list_bundles
 list_repositories()
 ['wildboar']
 list_bundles("wildboar")
 ['ucr', 'ucr-tiny', ... (and more)]

.. note::

    Repositories are cached locally in a folder controlled by the parameter ``cache_dir``. The default directory
    depends on platform. To change the default cache-directory:

    .. code-block:: python

     load_dataset("Wafer", repository="wildboar/ucr", cache_dir="/data/my_cache_drive")

To force re-download of an already cached repository set the parameter ``force`` to ``True``.

.. note::

    A wildboar repository string is composed of 2 mandatory and two optional
    components written as ``{repository}/{bundle}[:{version}][:{tag}]``

    ``{repository}``
       The repository identifier. List available bundles use ``list_bundles(repository)``.
       The identifier is composed of letters and match ``\w+``. List repositories
       with ``list_repositories()``.

    ``{bundle}``
       The bundle identifier, i.e., the dataset bundle of a repository. The available datasets
       can be listed with ``list_datasets("{repository}/{bundle}")``. The identifier
       is composed of alphanumeric characters and -, matching ``[a-zA-Z0-9\-]+``.

    ``{version}``
       The bundle version (defaults to the version specified by the repository). The version
       must match ``{major}[.{minor}][.{revision}]``.

    ``{tag}``
       The bundle tag (defaults to ``default``). The bundle tag is composed of
       letters and -, matching ``[a-zA-Z-]+``.

    **Examples**

    - ``wildboar/ucr``: the `ucr` bundle from the `wildboar` repository using the
      latest version and the ´default` tag.
    - ``wildboar/ucr-tiny:1.0``: the `ucr-tiny` bundle from the `wildboar` repository
      using the version `1.0` and `default` tag.
    - ``wildboar/outlier:1.0:hard``: the `outlier` bundle, with version `1.0`, from
      the `wildboar` repository using the tag `hard`.

Installing repositories
=======================

A repository implements the interface of the class ``wildboar.datasets.Repository``

.. note::

    The default wildboar-repository is implemented using a ``JSONRepository`` which
    specifies (versioned) datasets on a JSON endpoint.

Repositories are installed using the function ``install_repository`` which takes
either an url to a JSON-file or an instance of a ``Repository``.

.. code-block:: python

    from wildboar.datasets import install_repository
    install_repository("https://www.example.org/repo.json")
    list_repositories("example")
    load_dataset("example", repository="example/example")

Repositories can be refreshed using ``datasets.refresh_repositories()``.

Implementation details
======================

By default, repositories installed with ``install_repositoriy`` should point to a 
JSON-file, which describes the available datasets and the location where ``wildboar``
can download them. Below, we describe the format of the JSON-file:

.. code-block:: javascript

    {
        "name": "example", // required
        "version": "1.0",  // required
        "wildboar_requires": "1.0.4", // required, the minimum required wildboar version
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

- The attributes ``{key}``, ``{version}`` and ``{tag}`` in the ``bundle_url``
  are replaced with the bundle-key, bundle-version and bundle tag from the repository string. 
  All attribute are required in the url.

- The ``arrays`` attribute is optional. However, if it is not present, the dataset is
  assumed to be a single numpy array, where the last column contains the class label or 
  a numpy-dict with both ``x``, and ``y`` keys.

  - if any other value except ``x`` and/or ``y`` is present in the ``arrays``-list,
    it will be loaded as an ``extras``-dictionary and only returned if requested by the user.
  - if ``y`` is not present in arrays ``load_dataset`` return ``None`` for ``y``

- The ``bundles/version`` attribute, is the default version of the bundle which is used
  unless the user specifies an alternative version in the repository string.

- The ``bundles/tag`` attribute is the default tag of the bundle which is used unless 
  the user specifies an alternative bundle. If not specified, the tag is ``default``.

- The ``bundles/collections`` attribute is a dictionary of named collections of datasets
  which can be specified when using ``load_datasets(..., collection="key")``

The ``bundle_url`` points to a remote location that for each bundle ``key``, contains 
two files with extensions ``.zip`` and ``.sha`` respectively. 
In the example, ``bundle_url`` should contain the two files ``example/default-v1.0.zip``
and ``example/default-v1.0.sha`` The ``.sha``-file should contain the `sha1` hash of the
``.zip``-file to ensure the integrity of the downloaded file. The ``zip``-file should 
contain the datasets.

By default, wildboar supports dataset bundles formatted as ``zip``-files containing 
``npy`` or ``npz``-files, as created by ``numpy.save`` and ``numpy.savez``. 
The datasets in the ``zip``-file must be named according to ``{dataset_name}(_TRAIN|_TEST)?.(npy|npz)``. 
That is, the dataset name (as specified when using ``load_dataset``) and optionally 
``_TRAIN`` or ``_TEST`` followed by the extension ``npy`` or ``npz``. If there are multiple
datasets with the same name but different training or testing tags, they will be merged. 
As such, if both ``_TRAIN`` and ``_TEST`` files are present for the same name, ``load_dataset`` 
can return these train and test samples separately by setting ``merge_train_test=False``.

.. code-block:: python

    from wildboar.datasets import load_dataset
    x_train, x_test, y_train, y_test = load_dataset(
      'GunPoint', repository='wildboar/ucr', merge_train_test=False
    )