.. currentmodule:: wildboar

================
Distance metrics
================

Wildboar support both subsequence distance using :func:`distance.pairwise_subsequence_distance` 
and traditional distances using :func:`distance.pairwise_distance`. These and related functions
support different metrics, as specified by the ``metric`` argument and metric parameters
using the ``metric_params`` argument.

List of subsequence metrics
===========================
.. _list_of_subsequence_metrics:

.. list-table:: Subsequence distance metrics
   :widths: 20, 15, 25, 40
   :header-rows: 1

   * - Metric name
     - ``metric``
     - ``metric_params``
     - Comments

   * - Euclidean
     - ``"euclidean"``
     - ``{}``
     -

   * - Scaled Euclidean
     - ``"scaled_euclidean"``
     - ``{}``
     - Normalized
    
   * - Dynamic time warping
     - ``"dtw"``
     - ``{"r": float}``
     - Window ``r`` in ``[0, 1]``

   * - Scaled DTW
     - ``"scaled_dtw"``
     - ``{"r": float}``
     - Window ``r`` in ``[0, 1]``


List of metrics
===============
.. _list_of_metrics:

.. list-table:: Distance metrics
   :widths: 20, 15, 25, 40
   :header-rows: 1

   * - Metric name
     - ``metric``
     - ``metric_params``
     - Comments

   * - Euclidean
     - ``"euclidean"``
     - ``{}``
     -

   * - Dynamic time warping
     - ``"dtw"``
     - ``{"r": float}``
     - Window ``r`` in ``[0, 1]``. Elastic.

   * - Derivative DTW
     - ``"ddtw"``
     - ``{"r": float}``
     - Window ``r`` in ``[0, 1]``. Elastic.

   * - Weighted Derivative DTW
     - ``"wddtw"``
     - ``{"r": float, "g": float}``
     - Window ``r`` in ``[0, 1]``. Phase difference penelty ``g``. Elastic.

   