.. _examples-unsupervised:

============
Unsupervised
============

Outlier detection
=================

In this example, ``ShapeletIsolationForest`` is used to detect outliers. We
use the ``ShapeletForestEmbedding`` to visualize the time series and mark
each sample according to the outlier score assigned by the shapelet isolation forest.
True anomalous samples are encircled by a red circle and samples predicted as
anomalous are encircled by a black circle.

.. literalinclude:: code/outlier_isf.py

.. figure:: fig/outlier_isf.png

Matrix Profile
==============

The matrix profile is a data structure that annotates a time series with the distance
of the closest matching subsequence at the i:th index. In the first example, we join
every subsequence in the second sample with the first three samples.

.. literalinclude:: code/matrix_profile_ab.py

.. figure :: fig/matrix_profile_ab.png

In the second example, we self-join every subsequence with its closest position.

.. literalinclude:: code/matrix_profile.py

.. figure :: fig/matrix_profile.png



