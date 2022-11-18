# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

from cython.view cimport memoryview


cdef class Dataset:
    cdef memoryview data

    cdef readonly Py_ssize_t n_samples  # the number of samples
    cdef readonly Py_ssize_t n_timestep  # the number of timesteps
    cdef readonly Py_ssize_t n_dims

    cdef readonly Py_ssize_t sample_stride  # the stride for samples
    cdef readonly Py_ssize_t dim_stride  # the dimension stride

    cdef double* get_sample(self, Py_ssize_t i, Py_ssize_t dim) nogil