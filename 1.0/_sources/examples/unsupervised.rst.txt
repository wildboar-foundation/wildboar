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


