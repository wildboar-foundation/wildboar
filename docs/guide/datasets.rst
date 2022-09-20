========
Datasets
========
Wildboar is distributed with an advanced system for handling dataset repositories. A dataset repository can
be used to load benchmark datasets or to distribute or store datasets.

.. code-block:: python

    from wildboar.datasets import load_dataset
    x, y = load_dataset('GunPoint', repository='wildboar/ucr')

.. toctree::
   :maxdepth: 2
   :hidden:

   datasets/repositories
   datasets/preprocess
   datasets/filter
   datasets/outlier


