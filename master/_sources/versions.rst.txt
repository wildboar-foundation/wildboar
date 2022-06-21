==========
 Versions
==========

.. _version-1-1:

Version 1.1
===========

Highlights
----------

* New classifiers:

  - ``RocketClassifier``
  - ``RockestClassifier``
  - ``RandomShapeletClassifier``
  - ``IntervalTreeClassifier``
  - ``IntervalForestClassifier``
  - ``PivotTreeClassifier``
  - ``ProximityTreeClassifier``

* New regressors:

  - ``RocketRegressor``
  - ``RockestRegressor``
  - ``RandomShapeletRegressor``
  - ``IntervalTreeRegressor``
  - ``IntervalForestRegressor``

* New embeddings:

  - ``RocketEmbedding``
  - ``RandomShapeletEmbedding``
  - ``IntervalEmbedding``
  - ``PivotEmbedding``

* New annotations:

  - ``motifs``, find top motifs
  - ``regimes``, find regime changes

* New explanations:

  - ``IntervalImportance``, explain time series regions

* New features:

  - ``pairwise_subsequence_distance``
  - ``paired_subsequence_distance``
  - ``pairwise_distance``
  - ``paired_distance``
  - ``subsequence_match``
  - ``paired_subsequence_match``
  - ``matrix_profile``, including support for MASS distance

API Changes
-----------

* Rename ``datasets._filter`` to ``datasets.filter``.

* Rename the parameter ``shapelet`` of the ``tree_`` attribute to ``feature``.

Other improvements
------------------

* Binary builds for Apple ARM processors (M1 and M2)

* Drop support for Python 3.7 and introduce support for Python 3.10
