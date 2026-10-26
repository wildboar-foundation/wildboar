.. _glossary:

########
Glossary
########

Wildboar embraces the `glossary of terms by scikit-learn <sklearn:glossary>`__
with some additions.

time-series
   The most common input to Wildboar estimators. An array-like that for which
   :func:`numpy:numpy.asarray` will produce an array of appropriate shape, with
   rank 1, 2 or 3.

single time-series
   A 1d-array with shape ``(n_timestep, )`` or 2d-array with a single row or
   column.

univariate time-series
   A 2d-array with shape ``(n_samples, n_timestep)``.

multivariate time-series
   A 3d-array with shape ``(n_samples, n_dims, n_timestep)``.

variable-length time-series
   A time-series were each sample or dimension can be of different length. The
   maximum length is given by ``arr.shape[-1]``, but each sample can have a
   length shorter than that.

missing-values
   A missing value represented by :obj:`numpy.nan`.

end-of-series value
   A missing value that also indicates that the time-series is variable-length,
   represented by :obj:`wildboar.utils.variable_len.EoS`. Any value with an
   index larger than the first ``EoS`` is assumed not to be part of the series.
   :obj:`numpy.isnan` returns ``True`` for ``EoS``. To check for exactly
   ``EoS``, use :obj:`wildboar.utils.variable_len.is_end_of_series`.

timestep (or ``n_timestep``)
   The length of the time series given by ``arr.shape[-1]``.

dimensions (or ``n_dims``)
   The number of dimensions of a (multivariate) time series. For 2d-arrays the
   number of dimensions is 1 and for 3d-arrays the number of dimensions is
   given by ``arr.shape[1]``.

