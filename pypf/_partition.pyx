import numpy as np

from pypf._distribution import get_class_distribution

from libc.stdlib cimport malloc, free
cimport numpy as np
cimport cython

from libc.math cimport fabs, log2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float safe_info(Py_ssize_t n_classes,
                     double xw,
                     double[:] x,
                     double yw,
                     double[:] y,
                     double n_examples):
    cdef double x_sum = 0
    cdef double y_sum = 0
    cdef double xv, yv
    cdef Py_ssize_t i
    for i in range(n_classes):
        xv = x[i] / n_examples
        yv = y[i] / n_examples
        if xv > 0:
            x_sum += xv * log2(xv)
        if yv > 0:
            y_sum += yv * log2(yv)
    return fabs((xw / n_examples) * -x_sum + (yw / n_examples) * -y_sum)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def partition(long[:] idx, double[:] d, long[:] y, Py_ssize_t n_classes):
    if len(idx) == 1:
        return 0.0, d[0], idx, np.array([], dtype=int)

    cdef Py_ssize_t n_examples = idx.shape[0]
    cdef np.ndarray[Py_ssize_t] order = np.argsort(d)
    cdef double[:] left_d = np.zeros(n_classes)
    cdef double[:] right_d = np.zeros(n_classes)
    # cdef double* left_d_ptr = <double*>malloc(n_classes * sizeof(double))
    # cdef double[:] left_d = <double[:n_classes]> left_d_ptr

    # cdef double* right_d_ptr = <double*>malloc(n_classes * sizeof(double))
    # cdef double[:] right_d = <double[:n_classes]> right_d_ptr

    cdef Py_ssize_t example
    cdef Py_ssize_t cls
    for example in range(n_examples):
        right_d[y[idx[example]]] += 1.0
        
    # for cls in range(n_classes):
    #     left_d[cls] = 0.0

    cdef double prev_dist = d[order[0]]
    cdef Py_ssize_t prev_label = y[idx[order[0]]]

    cdef double lt_w = 1
    cdef double gt_w = n_examples - 1
    left_d[prev_label] += 1
    right_d[prev_label] -= 1

    cdef double entropy = safe_info(n_classes, lt_w, left_d, gt_w, right_d, n_examples)
    cdef double threshold = prev_dist
    cdef Py_ssize_t threshold_index = 1

    cdef Py_ssize_t i, order_i
    cdef double dist, e
    cdef Py_ssize_t label
    for i in range(1, n_examples):
        order_i = order[i]
        dist = d[order_i]
        label = y[idx[order_i]]
        if not label == prev_label:
            e = safe_info(n_classes, lt_w, left_d, gt_w, right_d, n_examples)
            if e < entropy:
#                print(e, lt_w, np.asarray(left_d), gt_w, np.asarray(right_d), n_examples)
                entropy = e
                threshold = (dist + prev_dist) / 2
                threshold_index = i

        prev_label = label
        prev_dist = dist

        lt_w += 1
        gt_w -= 1
        left_d[label] += 1
        right_d[label] -= 1
    
    # free(left_d_ptr)
    # free(right_d_ptr)
    return entropy, threshold, threshold_index, order

