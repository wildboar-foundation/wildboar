.. currentmodule:: wildboar.distance

.. _distance:

########
Distance
########

*****************
Pairwise distance
*****************

A typical workflow involves computing the distance between pairs of time series.
In Wildboar, we accomplish this using the :func:`pairwise_distance`-function
which computes the distance between each time series in `X` and optionally `Y`.
For example, we can compute the _Euclidean distance_ (default) between each
pair from time series of a single array as:

.. execute::
   :context:
   :show-return:

   from wildboar.datasets import load_two_lead_ecg
   from wildboar.distance import pairwise_distance
   X, y = load_two_lead_ecg()
   pairwise_distance(X[:3])

.. note::

  When computing the distance between time series in a single array, one can
  note that the upper and lower triangles of the distance matrix are mirror
  images of each other. Wildboar optimizes this and only computes the upper
  part of the triangle and mirrors it as the lower half, halving the
  computational cost. As such, it is advised to compute the distance as
  `pairwise_distance(X)` and not `pairwise_distance(X, X)`. Wildboar tries to
  be smart and computes the single triangle if `X is Y`, i.e., `X` and `Y` are
  the same object. However, it will not detect if `(X == Y).all()` (e.g.,
  `pairwise_distance(X, X.copy())`) and will compute the full matrix.

If we pass a second array, then the returned matrix is the pairwise distance
between the time series from both `X` and `Y`.

.. execute::
   :context:
   :show-return:

   pairwise_distance(X[:3], X[3:6])

By default, the :func:`pairwise_distance` function computes the Euclidean
distance. We can change the metric by specifying the `metric` parameter
`as one of the supported metrics <list_of_subsequence_metrics>`_:

.. execute::
   :context:
   :show-return:

   pairwise_distance(X[:3], X[3:6], metric="edr")

We can also specify optional extra parameters to the metric using the
`metric_params`-parameter:

.. execute::
   :context:
   :show-return:

   pairwise_distance(
      X[:3], X[3:6], metric="twe", metric_params={"stiffness": 0.3}
   )

We support a multitude of input configurations and the return value of
:func:`pairwise_distance` depends on the input shape(s) of `X` (with
`x_samples`) and `Y` (with `y_samples`) (and the `dim` parameter):

*A single 1d-array `X`*
  Returns the scalar $0$.

*A single 2d-array `X`*
  Returns a 2d-array with the distance between each sample in `X` of shape
  `(x_samples, x_samples)`.

  .. execute::
     :context:
     :show-return:

     pairwise_distance(X[0:2])

*A 1d-array `X` and a 1d-array `Y`*
  Returns a scalar with the distance of `X` to `Y`


  .. execute::
     :context:
     :show-return:

     pairwise_distance(X[0], X[2])

*A 1d-array `X` and a 2d-array `Y`*
  Returns a 1d-array of the distance between `X` and each sample in `Y` of shape
  `(y_samples, )`. If `Y` is a 1d-array and `X` is a 2d-array, a 1d-array, of
  shape `(x_samples, )`, with the distance between `Y` and each sample in `X` is
  returned.

  .. execute::
     :context:
     :show-return:

     pairwise_distance(X[0], X[2:5])

*A 2d-array `X` and a 2d-array `Y`*
  Returns a 2d-array with the distance between each sample in `X` and each
  sample in `Y` of shape `(x_samples, y_samples)`.

  .. execute::
     :context:
     :show-return:

     pairwise_distance(X[0:2], X[2:4])

*A 3d-array `X` and a 3d-array `Y`*
  Returns a 2d-array with the distance between each sample in `X` and each
  sample in `Y` of shape `(x_samples, y_samples)`.

  .. execute::
     :context:
     :show-return:

     x = X[0:6].reshape(2, 3, -1)
     y = X[6:12].reshape(2, 3, -1)
     pairwise_distance(x, y, dim="mean")

  If we set the parameter :python:`dim="full"` we return a 3d-array of shape
  :python:`(n_dims, n_samples, n_timestep)`. `Read more about
  multivariate support <multivariate_support>`_.

  .. execute::
     :context:
     :show-return:

     x = X[0:6].reshape(2, 3, -1)
     y = X[6:12].reshape(2, 3, -1)
     pairwise_distance(x, y, dim="full")


Paired distance
===============

Sometimes we do not need to compute the distance between every sample, but
instead the distance between _pairs_ of samples. For this purpose, we use the
:func:`paired_distance`, which accepts two arrays `X` and `Y` with shapes
:python:`(x_samples, x_timestep)` and :python:`(y_samples, y_timesteps)` with
:python:`x_timestep == y_timestep` for non-elastic metrics and `X` and `Y` that can be
broadcast to a uniform shape.

.. execute::
   :context:
   :show-return:

   from wildboar.distance import paired_distance
   paired_distance(X[0:3], X[3:6])

.. execute::
   :context:
   :show-return:

   paired_distance(X[0], X[3:6]) # Broadcasting

Similar to :func:`pairwise_distance`, we support the parameters `metric` and
`metric_params` accepting the same input:

.. execute::
   :context:
   :show-return:

   paired_distance(X[0], X[3:6], metric="wdtw", metric_params={"g": 0.1})

:func:`paired_distance` also supports a multitude of input configurations and
the output depends on that configuration:

*A 1d-array `X` and a 1d-array `Y`*
  Returns a scalar with the distance of `X` to `Y`.

*Two arrays that can be broadcast to the same shape*
  Returns a 1d-array of shape `(n_samples, )`.

  If 3d-arrays, and the parameter `dim` is set to `"full"`, we return a
  2d-array of shape :python:`(n_dims, n_samples)`. Refer to `more about
  multivariate support <multivariate_support>`_ for additional details.

  .. execute::
     :context:
     :show-return:

     x = X[0:6].reshape(2, 3, -1)
     y = X[6:12].reshape(2, 3, -1)
     paired_distance(x, y, dim="full")

****************
Minimum distance
****************

Another common task is to, given a query time series :math:`q` and a set of
time series :math:`\mathcal{T}` find the (`k`) time series in the set that has
the minimal distance to :math:`q`. A naive solution is to calculate the
distance between the query and every time series, selecting the `k` time series
with the lowest distance:

.. execute::
   :context:
   :show-return:

   pairwise_distance(X[0], X[6:12]).argmin()

Or if we want to return the `k` closest, we can use :func:`~numpy.argpartition`
(here ``k=4``):

.. execute::
   :context:
   :show-return:

   from numpy import argpartition

   argpartition(pairwise_distance(X[0:10], X[11:300]), 4)[:, :4]

Wildboar implements this pattern in the :func:`argmin_distance` function. The
function uses early abandoning to avoid computing the full distance for time
series that would never be among the closest.

.. execute::
   :context:
   :show-return:

   from wildboar.distance import argmin_distance

   argmin_distance(X[0:10], X[11:300], k=4)

The output of both calls are equivalent but the latter is roughly three times
faster for most metrics.

.. _lower_bound:

The :func:`argmin_distance` function accepts a parameter called
``lower_bound``, which takes an array containing pairwise lower bounds between
time series. The user can select the specific lower bounds to include in this
array, depending on needs or the specific distance measures. This flexibility
allows the user to tailor the estimation process to be as efficient as possible
by choosing an appropriate level of simplicity or complexity for the lower
bounds.

.. execute::
   :context:
   :show-return:

   from wildboar.distance.lb import DtwKeoghLowerBound

   lb_keogh = DtwKeoghLowerBound(r=0.1).fit(X[11:300])
   argmin_distance(
      X[0:10],
      X[11:300],
      metric="dtw",
      metric_params={"r": 0.1},
      lower_bound=lb_keogh.transform(X[0:10]),
   )

.. list-table:: Lower bounds and supported metric.
   :widths: 30 20 40
   :header-rows: 1

   * - Lower bound
     - Metric
     - Comment
   * - :class:`~wildboar.distance.lb.DtwKeoghLowerBound`
     - ``"dtw"``
     -
   * - :class:`~wildboar.distance.lb.DtwKimLowerBound`
     - ``"dtw"``
     -
   * - :class:`~wildboar.distance.lb.SaxLowerBound`
     - ``"euclidean"``
     - Requires z-normalization
   * - :class:`~wildboar.distance.lb.PaaLowerBound`
     - ``"euclidean"``
     - Requires z-normalization

.. _multivariate_support:

Multivariate support
====================

As described, both :func:`paired_distance` and :func:`pairwise_distance` support
multivariate time series by computing the *"interdimensional" distance* between
time series and (by default) reporting the *mean* (`dim="mean"`). Optionally, we
can return the full distance matrix by setting :python:`dim="full"`:

.. execute::
   :context:
   :show-return:

   x = X[0:6].reshape(2, 3, -1)
   y = X[6:12].reshape(2, 3, -1)
   pairwise_distance(x, y, dim="full")

By setting :python:`dim="full"`, Wildboar returns the full array of distances
between all dimensions. The returned array has the shape :python:`(n_dims,
x_samples, y_samples)`. Similarly, we can compute the paired distance:

.. execute::
   :context:
   :show-return:

   paired_distance(x, y, dim="full")

Note that the :func:`paired_distance` returns an array of shape
:python:`(n_dims, n_samples)`.

If we are interested in the distance between a single dimension we can either
slice the input data:

.. execute::
   :context:
   :show-return:

   d = pairwise_distance(x, y, dim="full")
   d[0]

or slice the full distance matrix:

.. execute::
   :context:
   :show-return:

   p = paired_distance(x, y, dim="full")
   p[0]

If we are **only** interested in a single dimension, we can set the `dim`
parameter to the dimension we are want:

.. execute::
   :context:
   :show-return:

   pairwise_distance(x, y, dim=0)

and for :func:`paired_distance`:

.. execute::
   :context:
   :show-return:

   paired_distance(x, y, dim=0)

By setting :python:`dim` to the desired dimension, we avoid computing the
distance between unwanted dimensions.

******************
Subsequence search
******************

Wildboar can also identify the (minimum) *subsequence* distance, i.e.,
:math:`\min\limits_{t'\in t}(s, t')`, where :math:`s` is a query and :math:`t`
is a time series, with :math:`|s| \le |t|`. Wildboar support both
:func:`pairwise_subsequence_distance` and :func:`paired_subsequence_distance`
which works similar to their non-subsequence counterparts, but also
:func:`subsequence_match` to identify subsequences with a distance within a
user specified _threshold_. Since Numpy does not support *jagged*-arrays, both
the paired and pairwise subsequence functions accept subsequences as ``list``
containing 1d-numpy arrays and 2d-numpy arrays with subsequences with the same
timestep.

Pairwise subsequence distance
=============================

Pairwise subsequence distance requires two inputs, the subsequences `Y`
(argument 1) and the time series `X` (argument 2):

.. execute::
   :context:
   :show-return:

   from wildboar.distance import pairwise_subsequence_distance

   pairwise_subsequence_distance(X[0, 30:60], X[6:12])


We can also pass multiple subsequences with the same number of time steps as a
2d-numpy array:

.. execute::
   :context:
   :show-return:

   pairwise_subsequence_distance(X[0:3, 30:60], X[6:12])

or with different number of time steps as a Python list:

.. execute::
   :context:
   :show-return:

   pairwise_subsequence_distance([X[0, 30:60], X[1, 0:10]], X[6:12])

Since we support multiple input configurations, the return value depends on the
inputs:

*A 1d-array `Y` and a 1d-array `X`*
  Returns a scalar with the minimum distance of `Y` to `X`.

*A 1d-array `Y` and a 2d or 3d-array `X`*
  Returns a 1d-array of shape `(n_samples, )` with the minimum distance of the
  subsequence to each sample.

*A 2d-array or list `Y` and a 1d-array `X`*
  Returns a 1d-array of shape `(n_subsequences, )` with the minimum distance of
  each subsequence to the sample.

*A 2d-array or list `Y` and a 2d or 3d-array `X`*
  Returns a 2d-array of shape `(n_subsequences, n_samples)` with the minimum
  distance of each subsequence to each sample.

.. note::

  :func:`pairwise_subsequence_distance` only supports
  univariate subsequences. As such, the `dim` parameter only accepts a `int`
  denoting the dimension in which to compute the subsequence distance. We can use
  the following code to compute the distance between the subsequence and each
  dimension of every sample, with consistent output.

  .. execute::
     :context:
     :show-return:

     def pairwise_sd_full(y, x):
         return np.stack(
             [pairwise_subsequence_distance(y, x, dim=dim) for dim in range(x.shape[1])],
             axis=0,
         )

     x = X[5:14].reshape(3, 3, -1)
     pairwise_sd_full(X[0, 30:60], x)

  .. execute::
     :context:
     :show-return:

     pairwise_sd_full([X[0,30:60], X[1, 10:20]], x)

We can request the *best matching index*, i.e., the index where the minimum
distance between the subsequence and the time series is identified, by setting
`return_index` to `True`:

.. code-block:: python

  >>> dist, idx = pairwise_subsequence_distance([X[0,30:60]], X[6:12],return_index=True)
  >>> dist
  array([1.66371456, 2.11914265, 1.13076667, 1.99043671, 1.73408875,
         1.84227457])
  >>> idx
  array([28, 30, 28, 34, 30, 28])

For example, we can see that the minimum distance of the subsequence is located
at index `28` for both the first, third and last sample.

*********
Benchmark
*********

As a component of the Wildboar test suite, we systematically evaluate the performance
of the estimators and, notably, the metrics, which are a crucial element of numerous
tasks. All Wildboar metrics, encompassing subsequence and elastic
metrics, are implemented in Cython to optimize CPU and memory efficiency.
Upon examining the relative performance of the various metrics and their
theoretical time and space complexity, it is evident that the elastic metrics are
generally two to three orders of magnitude less efficient than the non-elastic metrics.

Metrics
=======

.. csv-table::
  :file: ./metrics-benchmark.csv
  :header-rows: 1
  :widths: 40 12 12 12 12 12

Subsequence metrics
===================

.. csv-table::
  :file: ./subsequence_metric_benchmark.csv
  :header-rows: 1
  :widths: 40 12 12 12 12 12


.. [#lb] Uses lower-bounding as described by Rakthanmanon, T., Campana, B.,
   Mueen, A., Batista, G., Westover, B., Zhu, Q., Zakaria, J. and Keogh, E.,
   2012, August. Searching and mining trillions of time series subsequences
   under dynamic time warping. In Proceedings of the 18th ACM SIGKDD
   international conference on Knowledge discovery and data mining (pp.
   262-270). `See the UCR-suite for more details.
   <https://www.cs.ucr.edu/~eamonn/UCRsuite.html>`_

***************
Parallelization
***************

.. note::

   Wildboar employs parallelization to distribute the computation across
   multiple samples concurrently. Consequently, performance improvements may
   not be observed for a small number of extremely short time series. In
   certain cases, the additional time required to create threads may actually
   result in increased overall computation time.

.. warning::

   These measurements are from a Github CI build server and may not accurately
   reflect the actual performance gains.

All functions discussed thus far support parallelization. Parallelization can be enabled by
setting the `n_jobs` parameter to a positive integer corresponding to the number of cores to
utilize, or to ``-1`` to employ all available cores.

.. execute::
   :context:
   :show-return:

   import time
   start = time.perf_counter()
   pairwise_distance(X[:100], metric="dtw", n_jobs=-1)
   time.perf_counter() - start

Or we can specify an exact integer of the number of cores:

.. execute::
   :context:
   :show-return:

   import time
   start = time.perf_counter()
   pairwise_distance(X[:100], metric="dtw", n_jobs=1)
   time.perf_counter() - start

On my laptop, which is a MacBook equipped with an M2 processor, the computation
of pairwise distances using the parameter `n_jobs=-1` takes approximately 40
milliseconds, while performing the same computation with `n_jobs=1` takes
approximately 160 milliseconds.

