# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

from cython cimport view

ctypedef const double[:, :, ::view.contiguous] TSArray

cdef class Dataset:
    cdef TSArray view

    cdef readonly Py_ssize_t n_samples  # the number of samples
    cdef readonly Py_ssize_t n_timestep  # the number of timesteps
    cdef readonly Py_ssize_t n_dims

    cdef double* get_sample(self, Py_ssize_t i, Py_ssize_t dim) nogil