###########
Time series
###########

A time series is a (temporally) ordered sequence of real values. A univariate
time series has a single dimension, whereas a multivariate time series has
multiple dimensions. In Wildboar, time series are represented by Numpy-arrays.
A single univariate time series is represented as an array of shape
:python:`(1, n_timestep)` (or :python:`(n_timestep, )`). Wildboar represents a
multivariate time series as an array of shape :python:`(1, n_dims, n_timestep)`
(or :python:`(n_dims, n_timestep)` depending on context). A dataset of time
series is an array of ``n_samples``, i.e., for univariate time series an array
of shape :python:`(n_samples, n_timestep)` (or :python:`(n_samples, 1,
n_timestep)`) and a multivarete time series is represented as an array of shape
:python:`(n_samples, n_dims, n_timestep)`.

Most algorithms in wildboar assumes that the time series are of equal length and
without missing values. However, some datasets contain both missing values
and/or have time series or dimensions of unequal length. Wildboar represents the
`End-of-Sequence` identifier as :obj:`wildboar.eos` and *value is missing* by
:obj:`numpy.nan`. The ``EoS`` value is a valid IEEE754 ``NaN`` value, and will be
treated as :python:`True` by :obj:`numpy.isnan`, whereas :func:`wildboar.iseos`
will treat :obj:`numpy.nan` as :python:`False`.

.. note::
   By having ``EoS`` treated as ``NaN``, we can ignore it and just treat them
   as missing values.

.. code-block:: python

   >>> import numpy as np
   >>> import wildboar as wb
   >>> t1 = np.array([1, 2, 3, 1, 1, 1], dtype=float)
   >>> t2 = np.array([1, 2, 3, 1, wb.eos, wb.eos], dtype=float)
   >>> t3 = np.array([1, np.nan, 3, 3, 3, 3], dtype=float)
   >>> x = np.vstack([t1, t2, t3]) # dataset of (3, 6)

In the example, we create a dataset with 3 samples, where each sample has 6
timestep, and:

- :python:`x[0]` has no missing values and is of length ``n_samples``.
- :python:`x[1]` has no missing values and is of length 4
  (:python:`np.min(np.nonzero(wb.iseos(x[1]))[-1])`).
- :python:`x[2]` has a single missing value at index 2.

***************************
Variable length time series
***************************

.. warning::
   Support for variable length time series is not stable and the API will change
