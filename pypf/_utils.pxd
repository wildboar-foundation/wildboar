cimport numpy as np

cdef enum:
    RAND_R_MAX = 2147483647

cdef void intp_ndarray_to_size_t_ptr(np.ndarray[np.intp_t] i,
                                size_t* o)

cdef size_t label_distribution(const size_t* e, size_t n_samples,
                               const size_t* y, size_t n_classes,
                               double* dist) nogil

cdef void argsort(double* values, size_t* order, size_t length) nogil

cdef size_t rand_r(size_t* seed) nogil

cdef size_t rand_int(size_t min_val, size_t max_val, size_t* seed) nogil
