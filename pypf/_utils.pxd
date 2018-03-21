cimport numpy as np

cdef void intp_ndarray_to_size_t_ptr(np.ndarray[np.intp_t] i,
                                size_t* o)

cdef size_t label_distribution(size_t* e, size_t n_samples, size_t* y,
                               size_t n_classes, double* dist) nogil


