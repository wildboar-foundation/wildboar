==========
Supervised
==========

Shapelet forest vs. Nearest neighbors
=====================================

The following example show a comparison of the shapelet forest classifier
and the nearest neighbor classifier.

.. literalinclude:: code/classification_sf_vs_nn.py

.. figure:: fig/classification_sf_vs_nn.png

Univariate shapelet forest
==========================
The following example compare the AUC of the shapelet forest and the extra
shapelet trees classifier

.. literalinclude:: code/classification_sf_vs_est.py

Output

.. code-block:: python

   Classifier: Shapelet forest
    - fit-time:   1.07
    - test-score: 0.89
   Classifier: Extra Shapelet Trees
    - fit-time:   0.21
    - test-score: 0.86


Comparing classifiers
=====================
The following example compare the AUC of several classifiers over several datasets

.. literalinclude:: code/classification_cmp.py

.. csv-table:: Resulting AUC scores
   :file: tab/classification_cmp.csv
   :widths: 25, 25, 25, 25
   :header-rows: 1