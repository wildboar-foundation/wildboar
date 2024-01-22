.. currentmodule:: wildboar.distance

********
Distance
********

Pairwise distance
=================

A typical workflow involves computing the distance between pairs of time series.
In Wildboar, we accomplish this using the {func}`pairwise_distance`-function
which computes the distance between each time series in `X` and optionally `Y`.
For example, we can compute the _Euclidean distance_ (default if `metric=None`)
between each pair from time series of a single array as:

.. code-block:: python

  >>> from wildboar.datasets import load_two_lead_ecg
  >>> from wildboar.distance import pairwise_distance
  >>> X, y = load_two_lead_ecg()
  >>> pairwise_distance(X[:3])
  array([[0.        , 3.51158857, 5.11514381],
        [3.51158857, 0.        , 2.35905618],
        [5.11514381, 2.35905618, 0.        ]])

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

.. code-block:: python

  >>> pairwise_distance(X[:3], X[3:6])
  array([[4.85497117, 5.96086309, 6.18777928],
        [2.00606825, 5.23060212, 4.27419835],
        [1.64445581, 6.38965963, 4.79102936]])

By default, the :func:`pairwise_distance` function computes the Euclidean
distance. We can change the metric by specifying the `metric` parameter
`as one of the supported metrics <list_of_subsequence_metrics>`_:

.. code-block:: python

  >>> pairwise_distance(X[:3], X[3:6], metric="edr")
  array([[0.59756098, 0.47560976, 0.64634146],
        [0.08536585, 0.03658537, 0.13414634],
        [0.09756098, 0.25609756, 0.12195122]])

We can also specify optional extra parameters to the metric using the
`metric_params`-parameter:

.. code-block:: python

  >>> pairwise_distance(X[:3], X[3:6], metric="twe", metric_params={"stiffness": 0.3})
  array([[76.20881199, 73.62554784, 88.5536877 ],
        [27.49142159, 60.56024904, 50.24551102],
        [20.45513015, 81.60658533, 54.06099416]])

We support a multitude of input configurations and the return value of
:func:`pairwise_distance` depends on the input shape(s) of `X` (with
`x_samples`) and `Y` (with `y_samples`) (and the `dim` parameter):

*A single 1d-array `X`*
  Returns the scalar $0$.

*A single 2d-array `X`*
  Returns a 2d-array with the distance between each sample in `X` of shape
  `(x_samples, x_samples)`.

  .. code-block:: python

    >>> pairwise_distance(X[0:2])
    array([[0.        , 3.51158857],
          [3.51158857, 0.        ]])

*A 1d-array `X` and a 1d-array `Y`*
  Returns a scalar with the distance of `X` to `Y`


  .. code-block:: python
  
    >>> pairwise_distance(X[0], X[2])
    5.11514381

*A 1d-array `X` and a 2d-array `Y`*
  Returns a 1d-array of the distance between `X` and each sample in `Y` of shape
  `(y_samples, )`. If `Y` is a 1d-array and `X` is a 2d-array, a 1d-array, of
  shape `(x_samples, )`, with the distance between `Y` and each sample in `X` is
  returned.

  .. code-block:: python

    >>> pairwise_distance(X[0], X[2:5])
    array([5.11514381, 4.85497117, 5.96086309])

*A 2d-array `X` and a 2d-array `Y`*
  Returns a 2d-array with the distance between each sample in `X` and each
  sample in `Y` of shape `(x_samples, y_samples)`.

  .. code-block:: python
  
    >>> pairwise_distance(X[0:2], X[2:4])
    array([[5.11514381, 4.85497117],
          [2.35905618, 2.00606825]])

*A 3d-array `X` and a 3d-array `Y`*
  Returns a 2d-array with the distance between each sample in `X` and each
  sample in `Y` of shape `(x_samples, y_samples)`.

  .. code-block:: python

    >>> x = X[0:6].reshape(2, 3, -1)
    >>> y = X[6:12].reshape(2, 3, -1)
    >>> pairwise_distance(x, y)
    array([[5.48683192, 6.60301954],
          [4.34083722, 6.35954558]])
  .. note:: 
    If we set the parameter :python:`dim="full"` we return a 3d-array of shape
    :python:`(n_dims, n_samples, n_timestep)`. `Read more about
    multivariate support <multivariate_support>`_.

Paired distance
===============

Sometimes we do not need to compute the distance between every sample, but
instead the distance between _pairs_ of samples. For this purpose, we use the
:func:`paired_distance`, which accepts two arrays `X` and `Y` with shapes
:python:`(x_samples, x_timestep)` and :python:`(y_samples, y_timesteps)` with
:python:`x_timestep == y_timestep` for non-elastic metrics and `X` and `Y` that can be
broadcast to a uniform shape.

.. code-block:: python

  >>> from wildboar.distance import paired_distance
  >>> paired_distance(X[0:3], X[3:6])
  array([4.85497117, 5.23060212, 4.79102936])
  >>> paired_distance(X[0], X[3:6]) # Broadcasting
  array([4.85497117, 5.96086309, 6.18777928])

Similar to :func:`pairwise_distance`, we support the parameters `metric` and
`metric_params` accepting the same input:

.. code-block:: python

  >>> paired_distance(X[0], X[3:6], metric="wdtw", metric_params={"g": 0.1})
  array([0.50816474, 0.3299048 , 0.55193242])

:func:`paired_distance` also supports a multitude of input configurations and
the output depends on that configuration:

*A 1d-array `X` and a 1d-array `Y`*
  Returns a scalar with the distance of `X` to `Y`.

*Two arrays that can be broadcast to the same shape*
  Returns a 1d-array of shape `(n_samples, )`.

  .. note:: 
    If we set the parameter `dim="full"` we return a 2d-array of shape
    :python:`(n_dims, n_samples)`. Refer to `more about multivariate support
    <multivariate_support>`_ for additional details.

.. _multivariate_support:

Multivariate support
====================

As described, both :func:`paired_distance` and :func:`pairwise_distance` support
multivariate time series by computing the *"interdimensional" distance* between
time series and (by default) reporting the *mean* (`dim="mean"`). Optionally, we
can return the full distance matrix by setting :python:`dim="full"`:

.. code-block:: python

  >>> x = X[0:6].reshape(2, 3, -1)
  >>> y = X[6:12].reshape(2, 3, -1)
  >>> pairwise_distance(x, y, dim="full")
  array([[[5.48683192, 6.60301954],
          [4.34083722, 6.35954558]],

        [[2.50507001, 0.90920635],
          [5.27646127, 4.60041068]],

        [[3.60786006, 3.75645164],
          [6.26677146, 7.24823344]]])

By setting :python:`dim="full"`, Wildboar returns the full array of distances
between all dimensions. The returned array has the shape :python:`(n_dims,
x_samples, y_samples)`. Similarly, we can compute the paired distance:

.. code-block:: python

  >>> paired_distance(x, y, dim="full")
  array([[5.48683192, 6.35954558],
        [2.50507001, 4.60041068],
        [3.60786006, 7.24823344]])

Note that the :func:`paired_distance` returns an array of shape
:python:`(n_dims, n_samples)`.

If we are interested in the distance between a single dimension we can either
slice the input data or slice the full distance matrix:

.. code-block:: python

  >>> d = pairwise_distance(x, y, dim="full")
  >>> d[0]
  array([[5.48683192, 6.60301954],
        [4.34083722, 6.35954558]])
  >>> p = paired_distance(x, y, dim="full")
  >>> p[0]
  array([5.48683192, 6.35954558])

If we are **only** interested in a single dimension, we can set the `dim`
parameter to the dimension we are want:

.. code-block:: python

  >>> pairwise_distance(x, y, dim=0)
  array([[5.48683192, 6.60301954],
        [4.34083722, 6.35954558]]
  >>> paired_distance(x, y, dim=0)
  array([5.48683192, 6.35954558])

By setting :python:`dim` to the desired dimension, we avoid computing the
distance between unwanted dimensions.

******************
Subsequence search
******************


Wildboar can also identify the (minimum) *subsequence* distance, i.e.,
:math:`\min\limits_{t'\in t}(s, t')`, where :math:`s` is a query and :math`t`
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

.. code-block:: python
   
   >>> from wildboar.distance import pairwise_subsequence_distance
   >>> X, y = load_dataset("TwoLeadECG")
   >>> pairwise_subsequence_distance(X[0, 30:60], X[6:12])
   array([1.66371456, 2.11914265, 1.13076667, 1.99043671, 1.73408875,
          1.84227457])


We can also pass multiple subsequences with the same number of timesteps as a
2d-numpy array:

.. code-block:: python

  >>> pairwise_subsequence_distance(X[0:3, 30:60], X[6:12])
  array([[1.66371456, 1.2028058 ],
         [2.11914265, 0.85972633],
         [1.13076667, 0.85367621],
         [1.99043671, 0.86957415],
         [1.73408875, 0.64041732],
         [1.84227457, 1.33156061]])

or with different number of timesteps as a Python list:

.. code-block:: python

  >>> pairwise_subsequence_distance([X[0, 30:60], X[1, 0:10]], X[6:12])
  array([[1.66371456, 0.56698045],
         [2.11914265, 0.99489626],
         [1.13076667, 0.6790517 ],
         [1.99043671, 0.16754772],
         [1.73408875, 0.10973127],
         [1.84227457, 0.50583639]])

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

  .. code-block:: python

    >>> def pairwise_sd_full(y, x):
    ...    return np.stack(
    ...        [pairwise_subsequence_distance(y, x, dim=dim) for dim in range(x.shape[1])],
    ...        axis=0,
    ...    )
    ...
    >>> x = X[5:14].reshape(3, 3, -1)
    >>> pairwise_sd_full(X[0, 30:60], x)
    array([[2.21688671, 1.13076667, 1.84227457],
           [1.66371456, 1.99043671, 1.83210644],
           [2.11914265, 1.73408875, 1.50884094]])
    >>> pairwise_sd_full([X[0, 30:60], X[1, 10:20]], x)
    array([[[2.21688671, 0.18507116],
            [1.13076667, 0.11177626],
            [1.84227457, 0.15611733]],

           [[1.66371456, 0.21780536],
            [1.99043671, 0.13350353],
            [1.83210644, 0.09710811]],
  
           [[2.11914265, 0.75114125],
            [1.73408875, 0.13489775],
            [1.50884094, 0.09806374]]])

We can request the *best matching index*, i.e., the index where the minimum
distance between the subsequence and the time series is identified, by setting
`return_index` to `True`:

.. code-block:: python

  >>> dist, idx = pairwise_subsequence_distance([X[0, 30:60]], X[6:12], return_index=True)
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

As part of the Wildboar test suite, we continuously investigate the performance
of the estimators and, in particular since they are an integrate part of many
tasks, the metrics. All Wildboar metrics, including subsequence and elastic
metrics, are implemented in Cython for minimum CPU and memory utilization.
Inspecting the relative performance of the different metrics and their
theoretical time and space-complexity, we can see that the elastic metrics are
typically two to three orders of magnitudes slower than the non-elastic metrics.

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
