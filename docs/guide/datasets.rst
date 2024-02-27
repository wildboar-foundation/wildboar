.. currentmodule:: wildboar
.. _guide-datasets:

########
Datasets
########

Wildboar is distributed with an advanced system for handling dataset
repositories. A dataset repository can be used to load benchmark datasets or to
distribute or store datasets.

In its simplest for we can use the function :func:`datasets.load_dataset`:

.. code-block:: python

  from wildboar.datasets import load_dataset
  x, y = load_dataset('GunPoint', repository='wildboar/ucr')

****************
Loading datasets
****************

As described previously, :func:`~datasets.load_dataset` is the main entry point
for easy loading of datasets, but we can also iteratively load multiple datasets
using :func:`~datasets.load_datasets`. Currently, Wildboar only installs one
repository by default, the `wildboar` repository. We hope that others will find
the feature useful, and will distribute their datasets as Wildboar repositories.

.. note::

  One drawback of the current distribution approach is that we have to download
  the full bundle to load a single dataset. We hope to improve this in the
  future and download assets on-demand.

For small experiments, we can load a small selection of datasets from the
``wildboar/ucr-tiny`` bundle, either using :func:`~datasets.load_dataset` or
using one of the named functions, e.g., :func:`~datasets.load_gun_point`
(browse :mod:`wildboar.datasets` for all such functions).

Loading a single dataset
========================

We can load a single dataset as follows:

.. code-block:: python

   >>> from wildboar.datasets import load_dataset
   >>> x, y = load_dataset("GunPoint", repository="wildboar/ucr-tiny")
   Downloading ucr-tiny-v1.0.2-default.zip (688.43 KB)
      |██████████████████████████████████████████████----| 668.43/688.43 KB
   >>> x.shape
   (200, 150)

Wildboar offers additional operations that we can perform while loading
datasets, for example, we can
:doc:`preprocess the time series </guide/datasets/preprocess>` or return
optional training/testing parts by setting ``merge_train_test`` to
:python:`False`.

.. code-block:: python

   >>> x_train, x_test, y_train, y_test = load_dataset("GunPoint", merge_train_test=False)
   >>> x_train.shape, x_test.shape
   ((50, 150), (150, 150))

We can also force a re-download of an already cached bundle by setting `force`
to {python}`True`, and changing the `dtype` of the returned time series:

.. code-block:: python

   >>> load_datasets("GunPoint", dtype=float, force=True)
   # ... re-download dataset

.. note::

   To reduce the download size, the datasets downloaded from the
   ``wildboar``-repository are 32-bit floating point values. However,
   :func:`~datasets.load_dataset` converts the values to 64-bit when
   loading the data to conform with the default value conventions of
   Wildboar.

Loading multiple datasets
=========================

When running experiments, a common workflow is to load multiple dataset, fit
and evaluate some estimator. In Wildboar, we can repeatedly load datasets from
a bundle using the :func:`~datasets.load_datasets`-function:

.. code-block:: python

   >>> from wildboar.datasets import load_datasets
   >>> for name, (x, y) in load_datasets("wildboar/ucr-tiny"):
   ...     print(name, x.shape)
   ...
   Beef (60, 470)
   Coffee (56, 286)
   GunPoint (200, 150)
   SyntheticControl (600, 60)
   TwoLeadECG (1162, 82)

Loading multiple datasets also support setting the ``merge_train_test`` to
:python:`False`:

.. code-block:: python

   >>> for name, (x_train, x_test, y_train, y_test) in load_datasets("wildboar/ucr-tiny"):
   ...     print(name, x_train.shape)
   ...

.. _datasets_filter:

Filters
=======

We can also specify filters to filter the datasets on the number of dimensions,
samples, timesteps, labels and dataset names. We specify filters with the
`filter` parameter, which accepts a :python:`list`, :python:`dict` or
:python:`str`. We express string filters as:

::

                        ┌── Operator specification
            ┌───────────┴───────────┐
   (attribute)[<|<=|>|>=|=|~=](\d+|\w+)
   └────┬────┘└───────┬──────┘└───┬───┘
      │             │           └── A number or (part of) a dataset name
      │             └── The comparision operator
      └── The attribute name

The attribute name is one of (the self-explanatory) attributes:

``n_samples`` (:python:`int`)
   The number of samples.

``n_timesteps`` (:python:`int`)
   The number of time steps.

``n_dims`` (:python:`int`)
   The number of dimensions.

``n_labels`` (:python:`int`)
   The number of labels

``dataset`` (:python:`str`)
   The dataset name

The comparison operators for :python:`int` are ``<``, ``<=``, ``>``, ``>=`` and
``=``, for *less-than*, *less-than-or-equal*, *greater-than*,
*greater-than-or-equal* and *exactly-equal-to* respectively. The :python:`str`
comparison operators are ``=`` and ``~=``, for *exactly-equal-to* and
*exists-in* respectively.

Filters can be chained to support `and-also` using a `list` or a `dict`:

.. code-block:: python

   >>> large = "n_samples>=100"
   >>> large_multivariate = ["n_samples>=100", "n_dims>1"]
   >>> large_multiclass = {
   ...     "n_samples": ">=100",
   ...     "n_labels": ">2",
   ... }
   >>> load_datasets("wildboar/ucr-tiny", filter=large_multiclass)
   <generator object load_datasets at 0x7f262ce95d00>

.. warning::
   If we load multiple datasets with the parameter `merge_train_test` set to
   `False` filters are applied to the **training** part only.

:func:`~datasets.load_datasets` also accepts all parameters that are valid for
:func:`~datasets.load_dataset`, so we can also preprocess the time series:

.. code-block:: python

   >>> load_datasets("wildboar/ucr-tiny", filter=large, preprocess="minmax_scale")
   <generator object load_datasets at 0x7f262ce95d00>

.. toctree::
   :maxdepth: 2
   :hidden:

   datasets/repositories
   datasets/preprocess
