===========
Time series
===========

A time series is a (temporally) ordered sequence of real values. A univariate time
series has a single dimension, whereas a multivariate time series has multiple
dimensions. In wildboar, time series are represented by Numpy-arrays. A single
univariate time series is represented as an array of shape ``(1, n_timestep)`` (or
``(n_timestep, )``). A multivariate time series is represented as an array of shape
``(1, n_dims, n_timestep)`` (or ``(n_dims, n_timestep)`` depending on context). A
dataset of time series is an array of ``n_samples``, i.e., for univariate time series
an array of shape ``(n_samples, n_timestep)`` (or ``(n_samples, 1, n_timestep)``) and a
multivaraite time series is represented as an array of shape ``(n_samples, n_dims,
n_timestep)``.

Most algorithms in wildboar assumes that the time series are of equal length and without
missing values. However, some datasets contain both missing values and/or have time
series or dimensions of unequal length. In wildboar, the `end-of-sequence` in wildboar
is represented as ``-np.inf`` and `value-missing` by ``np.nan``.

.. code-block:: python

    >>> import numpy as np
    >>> t1 = np.array([1, 2, 3, 1, 1, 1], dtype=float)
    >>> t2 = np.array([1, 2, 3, 1, -np.inf, -np.inf], dtype=float)
    >>> t3 = np.array([1, np.nan, 3, 3, 3, 3], dtype=float)
    >>> x = np.vstack([t1, t2, t3]) # dataset of (3, 6)

In the example, we create a dataset with 3 samples, where each sample has 6 timestep.
``x[0]`` has no missing values and is of length n_samples. ``x[1]`` has no missing
values and is of length 4 (``np.min(np.nonzero(np.isneginf(x[1]))[-1])``). ``x[1]`` has 
a single missing value at index 2.
