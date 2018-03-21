# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
cimport cython

cdef void intp_ndarray_to_size_t_ptr(np.ndarray[np.intp_t] i, size_t* o):
    cdef size_t size = i.shape[0]
    cdef size_t p
    for p in range(size):
        o[p] = i[p]
        

cdef size_t label_distribution(size_t* e,
                               size_t n_samples,
                               size_t* y,
                               size_t n_classes,
                               double* dist) nogil:
    """ Computes the label distribution

    :return: number of classes included in the sample
    """
    cdef double weight = 1.0 / n_samples
    cdef size_t i, j, n_pos = 0
    for i in range(n_samples):
        j = e[i]
        dist[y[j]] += weight

    for i in range(n_classes):
        if dist[i] > 0:
            n_pos += 1
        
    return n_pos

