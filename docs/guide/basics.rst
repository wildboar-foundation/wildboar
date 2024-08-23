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

Most algorithms in wildboar assumes that the time series are of equal length
and without missing values. However, some datasets contain both missing values
and/or have time series or dimensions of unequal length. Wildboar represents
the `End-of-Sequence` identifier as :obj:`~wildboar.utils.variable_len.EOS` and
*value is missing* by :obj:`nupy.nan`. The ``EoS`` value is a valid IEEE754
``NaN`` value, and will be treated as :python:`True` by :obj:`numpy.isnan`,
whereas :func:`~wildboar.utils.variable_len.is_end_of_series` will treat
:obj:`numpy.nan` as :python:`False`.

.. note::
   By having ``EoS`` treated as ``NaN``, we can ignore it and just treat them
   as missing values.

.. execute::
   :show-return:

   import numpy as np
   from wildboar.utils.variable_len import EOS
   t1 = np.array([1, 2, 3, 1, 1, 1], dtype=float)
   t2 = np.array([1, 2, 3, 1, EOS, EOS], dtype=float)
   t3 = np.array([1, np.nan, 3, 3, 3, 3], dtype=float)
   x = np.vstack([t1, t2, t3])  # dataset of (3, 6)
   x

In the example, we construct a dataset comprising 3 samples, each containing 6
timesteps, where:

- :python:`x[0]` contains no missing values and has a sequence length equal to
  ``n_timestep``.
- :python:`x[1]` also contains no missing values but has a sequence length of
  4.
- :python:`x[2]` includes one missing value at the second index position.

***************************
Variable length time series
***************************

.. warning::
   Support for variable length time series is not stable and the API will change
