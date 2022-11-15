.. currentmodule:: wildboar

================
Distance metrics
================

Wildboar support both subsequence distance using :func:`distance.pairwise_subsequence_distance` 
and traditional distances using :func:`distance.pairwise_distance`. These and related functions
support different metrics, as specified by the ``metric`` argument and metric parameters
using the ``metric_params`` argument.

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

   * - Normalized Euclidean
     - ``"normalized_euclidean"``
     - ``{}``
     - Euclidean distance, where length has been scaled to have unit norm.
       Undefined cases result in 0.

   * - Scaled Euclidean
     - ``"scaled_euclidean"`` or ``"mass"``
     - ``{}``
     - Scales each subsequence to have zero mean and unit variance.
    
   * - Dynamic time warping
     - ``"dtw"``
     - ``{"r": float}``
     - Window ``r`` in ``[0, 1]``

   * - Scaled DTW
     - ``"scaled_dtw"``
     - ``{"r": float}``
     - Window ``r`` in ``[0, 1]``

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
    
   * - Normalized Euclidean
     - ``"normalized_euclidean"``
     - ``{}``
     - Euclidean distance, where length has been scaled to have unit norm.
       Undefined cases result in 0.

   * - Longest common subsequence
     - ``"lcss"``
     - ``{r: float, threshold: float}``
     - Window ``r`` in ``[0, 1]``.  ``threshold > 0.0``.

   * - Edit distance with real penalty
     - ``"erp"``
     - ``{r: float, g: float}``
     - Window ``r`` in ``[0, 1]``. Gap penalty ``g``.

   * - Edit distance for real sequences
     - ``"edr"``
     - ``{r: float, threshold: float}``
     - Window ``r`` in ``[0, 1]``. Match ``threshold``. Default ``1/4*max(std(x), std(y))``. Elastic.

   * - Dynamic time warping
     - ``"dtw"``
     - ``{"r": float}``
     - Window ``r`` in ``[0, 1]``. Elastic.

   * - Weighted DTW
     - ``"wdtw"``
     - ``{"r": float, "g": float}``
     - Window ``r`` in ``[0, 1]``. Phase difference penalty ``g``. Elastic.

   * - Derivative DTW
     - ``"ddtw"``
     - ``{"r": float}``
     - Window ``r`` in ``[0, 1]``. Elastic.

   * - Weighted Derivative DTW
     - ``"wddtw"``
     - ``{"r": float, "g": float}``
     - Window ``r`` in ``[0, 1]``. Phase difference penalty ``g``. Elastic.

   