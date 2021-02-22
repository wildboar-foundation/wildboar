========
Datasets
========
Wildboar is distributed with an advanced system for handling dataset repositories. A dataset repository can
be used to load benchmark datasets or to distribute or store datasets.

What is a repository?
=====================
I short, a repository is a collection of datasets bundles. More specifically, a repository links to bundles (zip-files) containing datasets
or dataset parts that can be downloaded, cached and loaded by wildboar.

How to use a repository?
========================
Repositories are either initialized directly or used together with the ``load_dataset`` function.

.. code-block:: python

    >>> from wildboar.datasets import load_dataset
    >>> x, y = load_dataset('GunPoint', repository='wildboar/ucr')
    # ... downloading repository to cache folder...
    >>> x.shape

Installed repositories and dataset bundles can be listed using the function
``list_repositories`` and ``list_bundles`` respectively.

.. code-block:: python

 >>> from wildboar.datasets import list_repositories, list_bundles
 >>> list_repositories()
 ['wildboar']
 >>> list_bundles("wildboar")
 ['ucr', 'ucr-tiny']

.. note::

    Repositories are cached locally in a folder controlled by the parameter ``cache_dir``. The default directory
    depends on platform. To change the default cache-directory:

    .. code-block:: python

     >>> load_dataset("Wafer", repository="wildboar/ucr", cache_dir="/data/my_cache_drive")

    .. warning::

        The default cache location changed in version 1.0.4. To use the old location set ``cache_dir``
        to ``'wildboar_cache'``

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

    >>> from wildboar.datasets import install_repository
    >>> install_repository("https://www.example.org/repo.json")
    >>> list_repositories("example")
    >>> load_dataset("example", repository="example/example")

Repository JSON specification
-----------------------------

The ``JSONRepository`` expects a JSON-file following the specification below.

.. code-block:: javascript

    {
        "name": "example",
        "version": "1.0",
        "wildboar_requires": "1.0.4",
        "bundle_url": "https://example.org/download/{key}-v{version}.zip",
        "bundles": [
          {
            "key": "example",
            "version": "1.0",
            "name": "UCR Time series repository",
            "description": "Example dataset",
            "class_index": -1
          },
        ]
    }