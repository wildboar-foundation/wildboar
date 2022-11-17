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

   * - Manhattan
     - ``"manhattan"``
     - ``{}``
     - 
    
   * - Minkowski
     - ``"minkowski"``
     - ``{}``
     - 

   * - Chebyshev
     - ``"chebyshev"``
     - ``{}``
     - 
    
   * - Cosine
     - ``"cosine"``
     - ``{}``
     - 

   * - Angular
     - ``"angular"``
     - ``{}``
     - 

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

   * - Manhattan
     - ``"manhattan"``
     - ``{}``
     - 
    
   * - Minkowski
     - ``"minkowski"``
     - ``{}``
     - 

   * - Chebyshev
     - ``"chebyshev"``
     - ``{}``
     - 
    
   * - Cosine
     - ``"cosine"``
     - ``{}``
     - 

   * - Angular
     - ``"angular"``
     - ``{}``
     - 

   * - Longest common subsequence [1]
     - ``"lcss"``
     - ``{r: float, threshold: float}``
     - Window ``r`` in ``[0, 1]``.  Match ``threshold``, default ``1``. Elastic.

   * - Edit distance with real penalty [2]
     - ``"erp"``
     - ``{r: float, g: float}``
     - Window ``r`` in ``[0, 1]``. Gap penalty ``g``, default ``0``. Elastic.

   * - Edit distance for real sequences [3]
     - ``"edr"``
     - ``{r: float, threshold: float}``
     - Window ``r`` in ``[0, 1]``. Match ``threshold``, default ``1/4*max(std(x), std(y))``. Elastic.

   * - Move-split-merge [4]
     - ``"msm"``
     - ``{r: float, c: float}``
     - Window ``r`` in ``[0, 1]``. Split/merge cost ``c``, default ``1``. Elastic.

   * - Time Warp Edit distance [5]
     - ``"twe"``
     - ``{r: float, edit_penalty: float, stiffness: float}``
     - Window ``r`` in ``[0, 1]``. Edit penalty (:math:`\lambda`), default ``1``.
       Stiffness (:math:`\nu`), default ``0.001``.

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


References
==========

[1] Hirschberg, D. (1977). 
  Algorithms for the longest common subsequence problem. 
  Journal of the ACM (JACM).

[2] Chen, L., & Ng, R. (2004). 
  On the Marriage of Lp-Norms and Edit Distance (30). 
  Proceedings of the Thirtieth International Conference on Very Large Data Base.

[3] Chen, L., Özsu, M. T., & Oria, V. (2005). 
  Robust and fast similarity search for moving object trajectories. 
  Proceedings from Proceedings of the International Conference on Management of Data

[4] Stefan, A., Athitsos, V., & Das, G. (2013). 
  The Move-Split-Merge Metric for Time Series. 
  IEEE Transactions on Knowledge and Data Engineering, 25(6), 1425-1438.

[5] Marteau, P.-F. (2008). 
  Time warp edit distance with stiffness adjustment for time series matching. 
  IEEE transactions on pattern analysis and machine intelligence, 31(2), 306-318.

