cimport cython

from libc.math cimport fabs, log2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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

