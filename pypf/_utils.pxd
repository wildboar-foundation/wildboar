cimport numpy as np

cdef enum:
    RAND_R_MAX = 2147483647

cdef size_t label_distribution(const size_t* samples,
                               size_t start,
                               size_t end,
                               const size_t* labels,
                               size_t n_classes,
                               double* dist) nogil

cdef void print_c_array_d(object name, double* arr, size_t length)

cdef void print_c_array_i(object name, size_t* arr, size_t length)

cdef void argsort(double* values, size_t* order, size_t length) nogil

cdef size_t rand_r(size_t* seed) nogil

cdef size_t rand_int(size_t min_val, size_t max_val, size_t* seed) nogil
