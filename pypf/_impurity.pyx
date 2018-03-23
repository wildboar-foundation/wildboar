# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport cython

from libc.math cimport fabs, log2

cpdef double safe_info(double xw,
                       double[:] x,
                       double yw,
                       double[:] y,
                       double n_examples) nogil:
    cdef double x_sum = 0
    cdef double y_sum = 0
    cdef double xv, yv
    cdef Py_ssize_t i
    for i in range(x.shape[0]):
        xv = x[i] / n_examples
        yv = y[i] / n_examples
        if xv > 0:
            x_sum += xv * log2(xv)
        if yv > 0:
            y_sum += yv * log2(yv)
    return fabs((xw / n_examples) * -x_sum + (yw / n_examples) * -y_sum)


cdef inline double info(double left_sum,
                        double* left_count,
                        double right_sum,
                        double* right_count,
                        size_t n_samples,
                        size_t n_labels) nogil:
    cdef double x_sum = 0
    cdef double y_sum = 0
    cdef double xv, yv
    cdef size_t i
    for i in range(n_labels):
        xv = left_count[i] / n_samples
        yv = right_count[i] / n_samples
        if xv > 0:
            x_sum += xv * log2(xv)
        if yv > 0:
            y_sum += yv * log2(yv)
    return fabs((left_sum / n_samples) * -x_sum + (right_sum / n_samples) * -y_sum)
