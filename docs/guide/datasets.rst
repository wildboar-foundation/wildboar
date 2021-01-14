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


In the default repository, wildboar bundles

wildboar/ucr
  UCR time series repository datasets optimized for loading and download speed.

wildboar/ucr-tiny
  A sample of datasets from the UCR time series repository

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

The format of the string to the ``repository`` parameter is ``{repository}/{bundle}``, where `repository`
is a name of a repository consisting of letters, numbers and dashes and `bundle` is the name of a dataset bundle
consisting of letters, numbers and dashes.

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






}