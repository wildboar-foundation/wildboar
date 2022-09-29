# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cimport numpy as np


cdef class CList:
    cdef list py_list
    cdef readonly Py_ssize_t size

    cdef void* get(self, Py_ssize_t i) nogil

cdef void strided_copy(Py_ssize_t stride, double* f, double* t, Py_ssize_t length) nogil

cdef void argsort(double *values, Py_ssize_t *order, Py_ssize_t length) nogil

cdef int realloc_array(void** a, Py_ssize_t p, Py_ssize_t size, Py_ssize_t *cap)  nogil except -1

cdef int safe_realloc(void** ptr, Py_ssize_t new_size) nogil except -1

cdef np.ndarray to_ndarray_int(Py_ssize_t *arr, Py_ssize_t n)

cdef np.ndarray to_ndarray_double(double *arr, Py_ssize_t n)