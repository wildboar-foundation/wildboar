# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False



from libc.math cimport fabs
from libc.math cimport log2

cdef inline double info(double left_sum,
                        double* left_count,
                        double right_sum,
                        double* right_count,
                         size_t n_labels) nogil:
    """Compute the entropy impurity

    :param left_sum: number of samples in the left partition
    :param left_count: number of samples per label in the left partition
    :param right_sum: number of samples in the right partiton
    :param right_count: number of samples per label in the right partition
    :param n_labels: the number of labels
    """
    cdef double n_samples = left_sum + right_sum
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

    return fabs((left_sum / n_samples) * -x_sum +
                (right_sum / n_samples) * -y_sum)
