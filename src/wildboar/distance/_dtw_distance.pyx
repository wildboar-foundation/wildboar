# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten

# This implementation is heavily inspired by the UCRSuite.
#
# References
#
#  - Rakthanmanon, et al., Searching and Mining Trillions of Time
#    Series Subsequences under Dynamic Time Warping (2012)
#  - http://www.cs.ucr.edu/~eamonn/UCRsuite.html

cimport numpy as np

import numpy as np

from libc.math cimport INFINITY, floor, sqrt
from libc.stdlib cimport free, malloc

from .._data cimport TSDatabase
from .._utils cimport fast_mean_std
from ._distance cimport DistanceMeasure, ScaledDistanceMeasure, TSCopy, TSView

from .._utils import check_array_fast


cdef void deque_init(Deque *c, Py_ssize_t capacity) nogil:
    c[0].capacity = capacity
    c[0].size = 0
    c[0].queue = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * capacity)
    c[0].front = 0
    c[0].back = capacity - 1


cdef void deque_reset(Deque *c) nogil:
    c[0].size = 0
    c[0].front = 0
    c[0].back = c[0].capacity - 1


cdef void deque_destroy(Deque *c) nogil:
    free(c[0].queue)


cdef void deque_push_back(Deque *c, Py_ssize_t v) nogil:
    c[0].queue[c[0].back] = v
    c[0].back -= 1
    if c[0].back < 0:
        c[0].back = c[0].capacity - 1

    c[0].size += 1


cdef void deque_pop_front(Deque *c) nogil:
    c[0].front -= 1
    if c[0].front < 0:
        c[0].front = c[0].capacity - 1
    c[0].size -= 1


cdef void deque_pop_back(Deque *c) nogil:
    c[0].back = (c[0].back + 1) % c[0].capacity
    c[0].size -= 1


cdef Py_ssize_t deque_front(Deque *c) nogil:
    cdef int tmp = c[0].front - 1
    if tmp < 0:
        tmp = c[0].capacity - 1
    return c[0].queue[tmp]


cdef Py_ssize_t deque_back(Deque *c) nogil:
    cdef int tmp = (c[0].back + 1) % c[0].capacity
    return c[0].queue[tmp]


cdef bint deque_empty(Deque *c) nogil:
    return c[0].size == 0


cdef Py_ssize_t deque_size(Deque *c) nogil:
    return c[0].size


cdef void find_min_max(
    Py_ssize_t offset,
    Py_ssize_t stride,
    Py_ssize_t length,
    double *T,
    Py_ssize_t r,
    double *lower,
    double *upper,
    Deque *dl,
    Deque *du,
) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t k

    cdef double current, prev

    deque_reset(du)
    deque_reset(dl)

    deque_push_back(du, 0)
    deque_push_back(dl, 0)

    for i in range(1, length):
        if i > r:
            k = i - r - 1
            upper[k] = T[offset + stride * deque_front(du)]
            lower[k] = T[offset + stride * deque_front(dl)]

        current = T[offset + stride * i]
        prev = T[offset + stride * (i - 1)]
        if current > prev:
            deque_pop_back(du)
            while (not deque_empty(du) and
                   current > T[offset + stride * deque_back(du)]):
                deque_pop_back(du)
        else:
            deque_pop_back(dl)
            while (not deque_empty(dl) and
                   current < T[offset + stride * deque_back(dl)]):
                deque_pop_back(dl)

        deque_push_back(du, i)
        deque_push_back(dl, i)

        if i == 2 * r + 1 + deque_front(du):
            deque_pop_front(du)
        elif i == 2 * r + 1 + deque_front(dl):
            deque_pop_front(dl)

    for i in range(length, length + r + 1):
        upper[i - r - 1] = T[offset + stride * deque_front(du)]
        lower[i - r - 1] = T[offset + stride * deque_front(dl)]

        if i - deque_front(du) >= 2 * r + 1:
            deque_pop_front(du)
        if i - deque_front(dl) >= 2 * r + 1:
            deque_pop_front(dl)


cdef inline double dist(double x, double y) nogil:
    cdef double s = x - y
    return s * s


cdef double constant_lower_bound(
    Py_ssize_t s_offset,
    Py_ssize_t s_stride,
    double *S,
    double s_mean,
    double s_std,
    Py_ssize_t t_offset,
    Py_ssize_t t_stride,
    double *T,
    double t_mean,
    double t_std,
    Py_ssize_t length,
    double best_dist,
) nogil:
    cdef double t_x0, t_y0, s_x0, s_y0
    cdef double t_x1, ty1, s_x1, s_y1
    cdef double t_x2, t_y2, s_x2, s_y2
    cdef double distance, min_dist
    if t_std == 0:
        return 0
    # first and last in T
    t_x0 = (T[t_offset] - t_mean) / t_std
    t_y0 = (T[t_offset + t_stride * (length - 1)] - t_mean) / t_std

    # first and last in S
    s_x0 = (S[s_offset] - s_mean) / s_std
    s_y0 = (S[s_offset + s_stride * (length - 1)] - s_mean) / s_std

    min_dist = dist(t_x0, s_x0) + dist(t_y0, s_y0)
    if min_dist >= best_dist:
        return min_dist

    t_x1 = (T[t_offset + t_stride * 1] - t_mean) / t_std
    s_x1 = (S[s_offset + s_stride * 1] - s_mean) / s_std
    min_dist += min(
        min(dist(t_x1, s_x0), dist(t_x0, s_x1)),
        dist(t_x1, s_x1))

    if min_dist >= best_dist:
        return min_dist

    t_y1 = (T[t_offset + t_stride * (length - 2)] - t_mean) / t_std
    s_y1 = (S[s_offset + s_stride * (length - 2)] - s_mean) / s_std
    min_dist += min(
        min(dist(t_y1, s_y1), dist(t_y0, s_y1)),
        dist(t_y1, s_y1))

    if min_dist >= best_dist:
        return min_dist

    t_x2 = (T[t_offset + t_stride * 2] - t_mean) / t_std
    s_x2 = (S[s_offset + s_stride * 2] - s_mean) / s_std
    min_dist += min(min(dist(t_x0, s_x2),
                        min(dist(t_x1, s_x2),
                            dist(t_x2, s_x2)),
                        dist(t_x2, s_x1)),
                    dist(t_x2, s_x0))

    if min_dist >= best_dist:
        return min_dist

    t_y2 = (T[t_offset + t_stride * (length - 3)] - t_mean) / t_std
    s_y2 = (S[s_offset + s_stride * (length - 3)] - s_mean) / s_std

    min_dist += min(min(dist(t_y0, s_y2),
                        min(dist(t_y1, s_y2),
                            dist(t_y2, s_y2)),
                        dist(t_y2, s_y1)),
                    dist(t_y2, s_y0))

    return min_dist


cdef double cumulative_bound(
    Py_ssize_t offset,
    Py_ssize_t stride,
    Py_ssize_t length,
    double mean,
    double std,
    double *T,
    double lu_mean,
    double lu_std,
    double *lower,
    double *upper,
    double *cb,
    double best_so_far,
) nogil:
    cdef double min_dist = 0
    cdef double x, d, us, ls
    cdef Py_ssize_t i

    for i in range(0, length):
        if min_dist >= best_so_far:
            break

        x = (T[offset + stride * i] - mean) / std
        us = (upper[i] - lu_mean) / lu_std
        ls = (lower[i] - lu_mean) / lu_std
        if x > us:
            d = dist(x, us)
        elif x < ls:
            d = dist(x, ls)
        else:
            d = 0

        min_dist += d
        cb[i] = d
    return min_dist


cdef inline double inner_scaled_dtw(
    Py_ssize_t s_offset,
    Py_ssize_t s_stride,
    int s_length,
    double s_mean,
    double s_std,
    double *S,
    double mean,
    double std,
    Py_ssize_t x_offset,
    double *X_buffer,
    int r,
    double *cb,
    double *cost,
    double *cost_prev,
    double min_dist,
) nogil:
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    cdef double x
    cdef double y
    cdef double z
    cdef double v
    cdef double min_cost, distance
    cdef bint std_zero = std == 0
    cdef bint s_std_zero = s_std == 0

    if std_zero and s_std_zero:
        return 0

    cdef double *cost_tmp
    for i in range(0, 2 * r + 1):
        cost[i] = INFINITY
        cost_prev[i] = INFINITY

    for i in range(0, s_length):
        k = max(0, r - i)
        min_cost = INFINITY
        for j in range(max(0, i - r), min(s_length, i + r + 1)):
            if i == 0 and j == 0:
                if not s_std_zero:
                    v = (S[s_offset] - s_mean) / s_std
                else:
                    v = 0

                if not std_zero:
                    v -= (X_buffer[x_offset] - mean) / std
                cost[k] = v * v
            else:
                if j - 1 < 0 or k - 1 < 0:
                    y = INFINITY
                else:
                    y = cost[k - 1]

                if i - 1 < 0 or k + 1 > 2 * r:
                    x = INFINITY
                else:
                    x = cost_prev[k + 1]

                if i - 1 < 0 or j - 1 < 0:
                    z = INFINITY
                else:
                    z = cost_prev[k]

                v = (S[s_offset + s_stride * i] - s_mean) / s_std
                if not std_zero:
                    v -= (X_buffer[x_offset + j] - mean) / std

                distance = v * v
                cost[k] = min(min(x, y), z) + distance

            if cost[k] < min_cost:
                min_cost = cost[k]

            k += 1

        if i + r < s_length - 1 and min_cost + cb[i + r + 1] >= min_dist:
            return min_cost + cb[i + r + 1]

        cost_tmp = cost
        cost = cost_prev
        cost_prev = cost_tmp
    return cost_prev[k - 1]


cdef double scaled_dtw_distance(
    Py_ssize_t s_offset,
    Py_ssize_t s_stride,
    Py_ssize_t s_length,
    double s_mean,
    double s_std,
    double *S,
    Py_ssize_t t_offset,
    Py_ssize_t t_stride,
    Py_ssize_t t_length,
    double *T,
    Py_ssize_t r,
    double *X_buffer,
    double *cost,
    double *cost_prev,
    double *s_lower,
    double *s_upper,
    double *t_lower,
    double *t_upper,
    double *cb,
    double *cb_1,
    double *cb_2,
    Py_ssize_t *index,
) nogil:
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef double lb_kim
    cdef double lb_k
    cdef double lb_k2

    cdef double ex = 0
    cdef double ex2 = 0
    cdef double tmp = 0

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t I
    cdef Py_ssize_t buffer_pos

    for i in range(t_length):
        current_value = T[t_offset + t_stride * i]
        ex += current_value
        ex2 += current_value * current_value

        buffer_pos = i % s_length
        X_buffer[buffer_pos] = current_value
        X_buffer[buffer_pos + s_length] = current_value

        if i >= s_length - 1:
            j = (i + 1) % s_length
            I = i - (s_length - 1)
            mean = ex / s_length
            tmp = ex2 / s_length - mean * mean
            if tmp > 0:
                std = sqrt(tmp)
            else:
                std = 1.0
            lb_kim = constant_lower_bound(s_offset, s_stride, S,
                                          s_mean, s_std, j, 1, X_buffer,
                                          mean, std, s_length, min_dist)

            if lb_kim < min_dist:
                lb_k = cumulative_bound(j, 1, s_length, mean, std, X_buffer,
                                        s_mean, s_std, s_lower, s_upper,
                                        cb_1, min_dist)
                if lb_k < min_dist:
                    lb_k2 = cumulative_bound(
                        s_offset, s_stride, s_length, s_mean, s_std, S,
                        mean, std, t_lower + I, t_upper + I, cb_2, min_dist)

                    if lb_k2 < min_dist:
                        if lb_k > lb_k2:
                            cb[s_length - 1] = cb_1[s_length - 1]
                            for k in range(s_length - 2, -1, -1):
                                cb[k] = cb[k + 1] + cb_1[k]
                        else:
                            cb[s_length - 1] = cb_2[s_length - 1]
                            for k in range(s_length - 2, -1, -1):
                                cb[k] = cb[k + 1] + cb_2[k]
                        dist = inner_scaled_dtw(
                            s_offset, s_stride, s_length, s_mean,
                            s_std, S, mean, std, j, X_buffer, r, cb,
                            cost, cost_prev, min_dist)

                        if dist < min_dist:
                            if index != NULL:
                                index[0] = (i + 1) - s_length
                            min_dist = dist

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return sqrt(min_dist)


cdef inline double inner_dtw(
    Py_ssize_t s_offset,
    Py_ssize_t s_stride,
    int s_length,
    double *S,
    Py_ssize_t t_offset,
    Py_ssize_t t_stride,
    double *T,
    int r,
    double *cost,
    double *cost_prev,
    double min_dist,
) nogil:
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    cdef double x
    cdef double y
    cdef double z
    cdef double v
    cdef double min_cost
    cdef double *cost_tmp

    for i in range(0, 2 * r + 1):
        cost[i] = INFINITY
        cost_prev[i] = INFINITY

    for i in range(0, s_length):
        k = max(0, r - i)
        for j in range(max(0, i - r), min(s_length, i + r + 1)):
            if i == 0 and j == 0:
                v = T[t_offset] - S[s_offset]
                cost[k] = v * v
            else:
                if j - 1 < 0 or k - 1 < 0:
                    y = INFINITY
                else:
                    y = cost[k - 1]

                if i - 1 < 0 or k + 1 > 2 * r:
                    x = INFINITY
                else:
                    x = cost_prev[k + 1]

                if i - 1 < 0 or j - 1 < 0:
                    z = INFINITY
                else:
                    z = cost_prev[k]

                v = T[t_offset + t_stride * i] - S[s_offset + s_stride * j]
                cost[k] = min(min(x, y), z) + v * v

            k += 1
        cost_tmp = cost
        cost = cost_prev
        cost_prev = cost_tmp
    return cost_prev[k - 1]


cdef double dtw_distance(
    Py_ssize_t s_offset,
    Py_ssize_t s_stride,
    Py_ssize_t s_length,
    double *S,
    Py_ssize_t t_offset,
    Py_ssize_t t_stride,
    Py_ssize_t t_length,
    double *T,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    Py_ssize_t *index,
) nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef Py_ssize_t i
    for i in range(t_length - s_length + 1):
        dist = inner_dtw(s_offset, s_stride, s_length, S,
                         t_offset + t_stride * i, t_stride, T,
                         r, cost, cost_prev, min_dist)
        if dist < min_dist:
            if index != NULL:
                index[0] = i
            min_dist = dist

    return sqrt(min_dist)


cdef double _dtw(
    Py_ssize_t x_offset,
    Py_ssize_t x_stride,
    Py_ssize_t x_length,
    double *X,
    double x_mean,
    double x_std,
    Py_ssize_t y_offset,
    Py_ssize_t y_stride,
    Py_ssize_t y_length,
    double *Y,
    double y_mean,
    double y_std,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
) nogil:
    """Dynamic time warping distance
    
    Parameters
    ----------
    x_offset : offset of x
    x_stride : stride of x
    x_length : length of x
    X : data of x
    x_mean : mean of array in x (if 0 ignored)
    x_std : std of array in x (or 1)
    y_offset : offset of y
    y_stride : stride of y
    y_length : length of y
    Y : data of y
    y_mean : mean of array in y (if 0 ignored)
    y_std : std of array in y (or 1)
    r : the warp window
    cost : cost matrix (max(x_length, y_length))
    cost_prev : cost matrix (max(x_length, y_length))

    Returns
    -------
    distance : the distance
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double x
    cdef double y
    cdef double z
    cdef double v

    v = (X[x_offset] - x_mean) / x_std
    v -= (Y[y_offset] - y_mean) / y_std

    cost_prev[0] = v * v
    for i in range(1, min(y_length, max(0, y_length - x_length) + r)):
        v = (X[x_offset] - x_mean) / x_std
        v -= (Y[y_offset + y_stride * i] - y_mean) / y_std
        cost_prev[i] = cost_prev[i - 1] + v * v

    if max(0, y_length - x_length) + r < y_length:
        cost_prev[max(0, y_length - x_length) + r] = INFINITY

    for i in range(1, x_length):
        j_start = max(0, i - max(0, x_length - y_length) - r + 1)
        j_stop = min(y_length, i + max(0, y_length - x_length) + r)
        if j_start > 0:
            cost[j_start - 1] = INFINITY
        for j in range(j_start, j_stop):
            x = cost_prev[j]
            if j > 0:
                y = cost[j - 1]
                z = cost_prev[j - 1]
            else:
                y = INFINITY
                z = INFINITY
            v = (X[x_offset + x_stride * i] - x_mean) / x_std
            v -= (Y[y_offset + y_stride * j] - y_mean) / y_std
            cost[j] = min(min(x, y), z) + v * v

        if j_stop < y_length:
            cost[j_stop] = INFINITY

        cost, cost_prev = cost_prev, cost
    return cost_prev[y_length - 1]


cdef void _dtw_align(double[:] X, double[:] Y, Py_ssize_t r, double[:,:] out) nogil:
    """Compute the warp alignment """
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start, j_stop
    cdef double v
    cdef double x, y, z
    v = X[0] - Y[0]
    out[0, 0] = v * v
    for i in range(1, min(X.shape[0], r + 1)):
        v = X[i] - Y[0]
        out[i, 0] = out[i - 1, 0] + v * v

    for i in range(1, min(Y.shape[0], max(0, Y.shape[0] - X.shape[0]) + r)):
        v = X[0] - Y[i]
        out[0, i] = out[0, i - 1] + v * v

    if max(0, Y.shape[0] - X.shape[0]) + r < Y.shape[0]:
        out[0, max(0, Y.shape[0] - X.shape[0]) + r] = INFINITY

    for i in range(1, X.shape[0]):
        j_start = max(0, i - max(0, X.shape[0] - Y.shape[0]) - r + 1)
        j_stop = min(Y.shape[0], i + max(0, Y.shape[0] - X.shape[0]) + r)
        if j_start > 0:
            out[i, j_start - 1] = INFINITY
        for j in range(j_start, j_stop):
            v = X[i] - Y[j]
            x = out[i - 1, j]
            y = out[i, j - 1] if j > 0 else INFINITY
            z = out[i - 1, j - 1] if j > 0 else INFINITY
            out[i, j] = min(min(x, y), z) + v * v

        if j_stop < Y.shape[0]:
            out[i, j_stop] = INFINITY


def _dtw_alignment(np.ndarray x, np.ndarray y, Py_ssize_t r, np.ndarray out=None):
    """Computes that dtw alignment matrix

    Parameters
    ----------
    x : ndarray of shape (x_size, )
        The first array

    y : ndarray of shape (y_size, )
        The second array

    r : int
        The warping window size

    out : ndarray of shape (x_size, y_size), optional
        To avoid allocating a new array out can be reused

    Returns
    -------
    alignment : ndarray of shape (x_size, y_size)
        - if `out` is given, alignment is out

        Contains the dtw alignment. Values outside the warping
        window size is undefined.
    """
    if not 0 < r < max(x.shape[0], y.shape[0]):
        raise ValueError("invalid r")
    x = check_array_fast(x, c_order=False)
    y = check_array_fast(y, c_order=False)
    cdef Py_ssize_t x_size = x.shape[0]
    cdef Py_ssize_t y_size = y.shape[0]
    if out is None:
        out = np.empty((x_size, y_size))

    if out.shape[0] < x_size or out.shape[1] < y_size:
        raise ValueError("out has wrong shape, got [%d, %d]" (x_size, y_size))

    _dtw_align(x, y, r, out)
    return out


def _dtw_distance(np.ndarray x, np.ndarray y, Py_ssize_t r, bint scale=False):
    """Compute the DTW of x and y

    Parameters
    ----------
    x : ndarray of shape (x_size)
        First time series

    y : ndarray of shape (y_size)
        Second time series

    r : int
        Size of warping window

    scale : bool, optional
        Standardize the arrays

    Returns
    -------
    dtw_distance, float
        The dynamic time warping distance

    Notes
    -----
    This implementation uses `2 * max(y_size, x_size)` memory.
    """
    if not 0 < r < max(x.shape[0], y.shape[0]):
        raise ValueError("invalid r")
    x = check_array_fast(x)
    y = check_array_fast(y)
    cdef Py_ssize_t x_length = <Py_ssize_t> x.shape[0]
    cdef Py_ssize_t x_stride = <Py_ssize_t> x.strides[0] / <Py_ssize_t> x.itemsize
    cdef double *x_data = <double*> x.data
    cdef Py_ssize_t y_length = <Py_ssize_t> y.shape[0]
    cdef Py_ssize_t y_stride = <Py_ssize_t> y.strides[0] / <Py_ssize_t> y.itemsize
    cdef double *y_data = <double*> y.data
    cdef double x_mean = 0
    cdef double x_std = 1
    cdef double y_mean = 0
    cdef double y_std = 1
    if scale:
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std == 0.0:
            x_std = 1.0

        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0.0:
            x_std = 1.0

    cdef double *cost = <double*> malloc(sizeof(double) * max(x_length, y_length))
    cdef double *cost_prev = <double*> malloc(sizeof(double) * max(x_length, y_length))
    cdef dist = _dtw(0, x_stride, x_length, x_data, x_mean, x_std,
                     0, y_stride, y_length, y_data, y_mean, y_std,
                     r, cost, cost_prev)
    free(cost)
    free(cost_prev)
    return sqrt(dist)


def _dtw_pairwise_distance(np.ndarray x, Py_ssize_t r):
    """Compute the distance between pairs of rows in x

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep)
        The time series

    r : int
        The warping window

    Returns
    -------
    pairwise_distance : ndarray of shape (n_samples, n_samples)
        The distances between pairs of samples (i, j)
    """
    if not 0 < r < x.shape[1]:
        raise ValueError("invalid r")
    x = check_array_fast(x, ensure_2d=True)
    cdef Py_ssize_t length = x.shape[1]
    cdef np.ndarray dists = np.empty((x.shape[0], x.shape[0]), dtype=np.float64)
    cdef Py_ssize_t i, j
    cdef Py_ssize_t i_offset, j_offset
    cdef Py_ssize_t sample_stride = <Py_ssize_t> x.strides[0] / <Py_ssize_t> x.itemsize
    cdef Py_ssize_t timestep_stride = <Py_ssize_t> x.strides[1] / <Py_ssize_t> x.itemsize
    cdef double *data = <double*> x.data
    cdef double *cost = <double*> malloc(sizeof(double) * length)
    cdef double *cost_prev = <double*> malloc(sizeof(double) * length)
    cdef double dist
    for i in range(x.shape[0]):
        i_offset = i * sample_stride
        dists[i, i] = 0
        for j in range(i + 1, x.shape[0]):
            j_offset = j * sample_stride
            dist = sqrt(_dtw(i_offset, timestep_stride, length, data, 0, 1,
                             j_offset, timestep_stride, length, data, 0, 1,
                             r, cost, cost_prev))
            dists[i, j] = dist
            dists[j, i] = dist

    free(cost)
    free(cost_prev)
    return dists


def _dtw_envelop(np.ndarray x, Py_ssize_t r):
    if not 0 < r < x.shape[0]:
        raise ValueError("invalid r")
    x = check_array_fast(x)

    cdef Deque du
    cdef Deque dl
    cdef Py_ssize_t length = x.shape[0]
    cdef Py_ssize_t stride = x.strides[0] / <Py_ssize_t> x.itemsize
    cdef double *data = <double*> x.data
    cdef np.ndarray lower = np.empty(length, dtype=np.float64)
    cdef np.ndarray upper = np.empty(length, dtype=np.float64)
    cdef double *lower_data = <double*> lower.data
    cdef double *upper_data = <double*> upper.data

    deque_init(&dl, 2 * r + 2)
    deque_init(&du, 2 * r + 2)
    find_min_max(0, stride, length, data, r, lower_data, upper_data, &dl, &du)

    deque_destroy(&dl)
    deque_destroy(&du)
    return lower, upper


def _dtw_lb_keogh(np.ndarray x, np.ndarray lower, np.ndarray upper, Py_ssize_t r):
    if not 0 < r < x.shape[0]:
        raise ValueError("invalid r")
    x = check_array_fast(x)
    lower = check_array_fast(lower)
    upper = check_array_fast(upper)
    cdef Py_ssize_t i
    cdef Py_ssize_t length = x.shape[0]
    cdef Py_ssize_t stride = x.strides[0] / <Py_ssize_t> x.itemsize
    cdef double *data = <double*> x.data
    cdef double *lower_data
    cdef double *upper_data
    if lower.strides[0] / <Py_ssize_t> lower.itemsize == 1:
        lower_data = <double*> lower.data
    else:
        lower_data = <double*> malloc(sizeof(double) * length)
        for i in range(length):
            lower_data[i] = lower[i]

    if upper.strides[0] / <Py_ssize_t> lower.itemsize == 1:
        upper_data = <double*> upper.data
    else:
        upper_data = <double*> malloc(sizeof(double) * length)
        for i in range(length):
            upper_data[i] = upper[i]
    cdef np.ndarray cb = np.empty(length, dtype=np.float64)
    cdef double *cb_data = <double*> cb.data
    cdef double min_dist
    min_dist = cumulative_bound(0, stride, length, 0, 1, data, 0, 1, lower_data, upper_data, cb_data, INFINITY)
    if <double*> upper.data != upper_data:
        free(upper_data)
    if <double*> lower.data != lower_data:
        free(lower_data)
    return sqrt(min_dist), cb


cdef inline Py_ssize_t _compute_warp_width(Py_ssize_t length, double r) nogil:
    # Warping path should be [0, length - 1]
    if r == 1:
        return length - 1
    if r < 1:
        return <Py_ssize_t> floor(length * r)
    else:
        return <Py_ssize_t> min(floor(r), length - 1)


cdef class ScaledDtwDistance(ScaledDistanceMeasure):
    cdef double *X_buffer
    cdef double *lower
    cdef double *upper
    cdef double *cost
    cdef double *cost_prev
    cdef double *cb
    cdef double *cb_1
    cdef double *cb_2

    cdef Deque du
    cdef Deque dl

    # The default and minimum size of the cost matrices
    cdef Py_ssize_t cost_size
    cdef double r


    def __cinit__(self, Py_ssize_t n_timestep, double r=0):
        super().__init__(n_timestep)
        if r < 0:
            raise ValueError("illegal warp width")
        self.r = r
        self.cost_size = _compute_warp_width(n_timestep, self.r) * 2 + 1
        self.X_buffer = <double*> malloc(sizeof(double) * n_timestep * 2)
        self.lower = <double*> malloc(sizeof(double) * n_timestep)
        self.upper = <double*> malloc(sizeof(double) * n_timestep)
        self.cost = <double*> malloc(sizeof(double) * self.cost_size)
        self.cost_prev = <double*> malloc(sizeof(double) * self.cost_size)
        self.cb = <double*> malloc(sizeof(double) * n_timestep)
        self.cb_1 = <double*> malloc(sizeof(double) * n_timestep)
        self.cb_2 = <double*> malloc(sizeof(double) * n_timestep)

        if (
            self.X_buffer == NULL or
            self.lower == NULL or
            self.upper == NULL or
            self.cost == NULL or
            self.cost_prev == NULL or
            self.cb == NULL or
            self.cb_1 == NULL or
            self.cb_2 == NULL
        ):
            raise MemoryError()

        deque_init(&self.dl, 2 * _compute_warp_width(n_timestep, self.r) + 2)
        deque_init(&self.du, 2 * _compute_warp_width(n_timestep, self.r) + 2)


    def __reduce__(self):
        return self.__class__, (self.n_timestep, self.r)


    def __dealloc__(self):
        free(self.X_buffer)
        free(self.lower)
        free(self.upper)
        free(self.cost)
        free(self.cost_prev)
        free(self.cb)
        free(self.cb_1)
        free(self.cb_2)
        deque_destroy(&self.dl)
        deque_destroy(&self.du)


    cdef int init_ts_view(
        self,
        TSDatabase *td_ptr,
        TSView *ts_view,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil:
        cdef TSDatabase td = td_ptr[0]
        ScaledDistanceMeasure.init_ts_view(
            self, td_ptr, ts_view, index, start, length, dim
        )

        cdef DtwExtra *dtw_extra = <DtwExtra*> malloc(sizeof(DtwExtra))
        dtw_extra[0].lower = <double*> malloc(sizeof(double) * length)
        dtw_extra[0].upper = <double*> malloc(sizeof(double) * length)

        cdef Py_ssize_t shapelet_offset = (
            index * td.sample_stride +
            start * td.timestep_stride +
            dim * td.dim_stride
        )
        cdef Py_ssize_t warp_width = _compute_warp_width(length, self.r)
        find_min_max(
            shapelet_offset,
            td.timestep_stride,
            length,
            td.data,
            warp_width,
            dtw_extra[0].lower,
            dtw_extra[0].upper,
            &self.dl,
            &self.du,
        )

        ts_view[0].extra = dtw_extra
        return 0


    cdef int init_ts_copy(self, TSCopy *shapelet, TSView *tv_ptr, TSDatabase *td_ptr) nogil:
        cdef int err = ScaledDistanceMeasure.init_ts_copy(
            self, shapelet, tv_ptr, td_ptr
        )
        if err == -1:
            return -1

        cdef TSDatabase td = td_ptr[0]
        cdef TSView shapelet_info = tv_ptr[0]

        cdef DtwExtra *dtw_extra = <DtwExtra*> malloc(sizeof(DtwExtra))
        cdef Py_ssize_t length = shapelet[0].length
        dtw_extra[0].lower = <double*> malloc(sizeof(double) * length)
        dtw_extra[0].upper = <double*> malloc(sizeof(double) * length)

        cdef Py_ssize_t warp_width = _compute_warp_width(length, self.r)
        find_min_max(
            0,
            1,
            length,
            shapelet[0].data,
            warp_width,
            dtw_extra[0].lower,
            dtw_extra[0].upper,
            &self.dl,
            &self.du,
        )
        shapelet[0].extra = dtw_extra
        return 0


    cdef int init_ts_copy_from_obj(self, TSCopy *tc, object obj):
        cdef int err = ScaledDistanceMeasure.init_ts_copy_from_obj(
            self, tc, obj
        )
        if err == -1:
            return -1
        dim, arr = obj
        cdef Py_ssize_t length = tc[0].length
        cdef DtwExtra *dtw_extra = <DtwExtra*> malloc(sizeof(DtwExtra))
        dtw_extra[0].lower = <double*> malloc(sizeof(double) * length)
        dtw_extra[0].upper = <double*> malloc(sizeof(double) * length)

        cdef Py_ssize_t warp_width = _compute_warp_width(length, self.r)
        find_min_max(
            0,
            1,
            length,
            tc[0].data,
            warp_width,
            dtw_extra[0].lower,
            dtw_extra[0].upper,
            &self.dl,
            &self.du,
        )
        tc[0].extra = dtw_extra
        return 0


    cdef double ts_copy_sub_distance(
        self,
        TSCopy *s,
        TSDatabase *td,
        Py_ssize_t t_index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        cdef Py_ssize_t sample_offset = (t_index * td.sample_stride + s.dim * td.dim_stride)
        cdef double *s_lower
        cdef double *s_upper
        cdef DtwExtra *extra
        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)

        if s.extra != NULL:
            extra = <DtwExtra*> s.extra
            s_lower = extra[0].lower
            s_upper = extra[0].upper
        else:
            s_lower = <double*> malloc(sizeof(double) * s.length)
            s_upper = <double*> malloc(sizeof(double) * s.length)

            find_min_max(
                0,
                1,
                s.length,
                s.data,
                warp_width,
                s_lower,
                s_upper,
                &self.dl,
                &self.du,
            )

        find_min_max(
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            warp_width,
            self.lower,
            self.upper,
            &self.dl,
            &self.du,
        )

        cdef double distance = scaled_dtw_distance(
            0,
            1,
            s.length,
            s.mean,
            s.std,
            s.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            warp_width,
            self.X_buffer,
            self.cost,
            self.cost_prev,
            s_lower,
            s_upper,
            self.lower,
            self.upper,
            self.cb,
            self.cb_1,
            self.cb_2,
            return_index,
        )
        if s.extra == NULL:
            free(s_lower)
            free(s_upper)

        return distance


    cdef double ts_view_sub_distance(self, TSView *s, TSDatabase *td, Py_ssize_t t_index) nogil:
        cdef Py_ssize_t sample_offset = (t_index * td.sample_stride + s.dim * td.dim_stride)
        cdef Py_ssize_t shapelet_offset = (
            s.index * td.sample_stride +
            s.dim * td.dim_stride +
            s.start * td.timestep_stride
        )

        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)
        cdef DtwExtra *dtw_extra = <DtwExtra*> s.extra
        find_min_max(
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            warp_width,
            self.lower,
            self.upper,
            &self.dl,
            &self.du,
        )

        return scaled_dtw_distance(
            shapelet_offset,
            td.timestep_stride,
            s.length,
            s.mean,
            s.std,
            td.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            warp_width,
            self.X_buffer,
            self.cost,
            self.cost_prev,
            dtw_extra[0].lower,
            dtw_extra[0].upper,
            self.lower,
            self.upper,
            self.cb,
            self.cb_1,
            self.cb_2,
            NULL,
        )


    cdef double ts_copy_distance(self, TSCopy *s, TSDatabase *td, Py_ssize_t t_index) nogil:
        cdef Py_ssize_t sample_offset = (t_index * td.sample_stride + s.dim * td.dim_stride)
        cdef Py_ssize_t warp_width = max(_compute_warp_width(s.length, self.r), 1)

        cdef Py_ssize_t max_length = max(s.length, td.n_timestep)
        if max_length > self.cost_size:
            free(self.cost)
            free(self.cost_prev)
            self.cost = <double*> malloc(sizeof(double) * max_length)
            self.cost_prev = <double*> malloc(sizeof(double) * max_length)
            if self.cost == NULL or self.cost_prev == NULL:
                with gil:
                    raise MemoryError()

        cdef double t_mean, t_std
        fast_mean_std(
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            &t_mean,
            &t_std,
        )
        if t_std == 0.0:
            t_std = 1.0

        cdef double dist = _dtw(
            0,
            1,
            s.length,
            s.data,
            s.mean, # assuming TSCopy is initialized with self
            s.std,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            t_mean,
            t_std,
            warp_width,
            self.cost,
            self.cost_prev
        )
        return sqrt(dist)

    cdef bint support_unaligned(self) nogil:
        return True


cdef class DtwDistance(DistanceMeasure):
    cdef double *cost
    cdef double *cost_prev
    cdef double r
    cdef Py_ssize_t cost_size


    def __cinit__(self, Py_ssize_t n_timestep, double r=0):
        super().__init__(n_timestep)
        if r < 0:
            raise ValueError("illegal warp width")
        self.r = r
        self.cost_size = _compute_warp_width(n_timestep, self.r) * 2 + 1
        self.cost = <double*> malloc(sizeof(double) * self.cost_size)
        self.cost_prev = <double*> malloc(sizeof(double) * self.cost_size)

        if self.cost == NULL or self.cost_prev == NULL:
            raise MemoryError()


    def __dealloc__(self):
        free(self.cost)
        free(self.cost_prev)


    def __reduce__(self):
        return self.__class__, (self.n_timestep, self.r)


    cdef double ts_copy_sub_distance(
        self,
        TSCopy *s,
        TSDatabase *td,
        Py_ssize_t t_index,
        Py_ssize_t *return_index = NULL,
    ) nogil:
        cdef Py_ssize_t sample_offset = (t_index * td.sample_stride + s.dim * td.dim_stride)
        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)

        return dtw_distance(
            0,
            1,
            s.length,
            s.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            warp_width,
            self.cost,
            self.cost_prev,
            return_index,
        )


    cdef double ts_view_sub_distance(
            self, TSView *ts_ptr, TSDatabase *td_ptr, Py_ssize_t t_index) nogil:
        cdef TSDatabase td = td_ptr[0]
        cdef TSView s = ts_ptr[0]
        cdef Py_ssize_t sample_offset = (t_index * td.sample_stride +
                                     s.dim * td.dim_stride)
        cdef Py_ssize_t shapelet_offset = (s.index * td.sample_stride +
                                       s.dim * td.dim_stride +
                                       s.start * td.timestep_stride)
        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)
        return dtw_distance(
            shapelet_offset,
            td.timestep_stride,
            s.length,
            td.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            warp_width,
            self.cost,
            self.cost_prev,
            NULL)


    cdef int ts_copy_matches(
        self,
        TSCopy *s,
        TSDatabase *td,
        Py_ssize_t t_index,
        double threshold,
        Py_ssize_t** matches,
        double** distances,
        Py_ssize_tn_matches,
    ) nogil except -1:
        with gil:
            raise NotImplemented()

    cdef double ts_copy_distance(self, TSCopy *s, TSDatabase *td, Py_ssize_t t_index) nogil:
        cdef Py_ssize_t sample_offset = (t_index * td.sample_stride + s.dim * td.dim_stride)
        cdef Py_ssize_t warp_width = max(_compute_warp_width(s.length, self.r), 1)
        cdef Py_ssize_t max_length = max(s.length, td.n_timestep)

        if max_length > self.cost_size:
            free(self.cost)
            free(self.cost_prev)
            self.cost = <double*> malloc(sizeof(double) * max_length)
            self.cost_prev = <double*> malloc(sizeof(double) * max_length)
            if self.cost == NULL or self.cost_prev == NULL:
                with gil: raise MemoryError()

        cdef double dist = _dtw(
            0,
            1,
            s.length,
            s.data,
            0,
            1,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            0,
            1,
            warp_width,
            self.cost,
            self.cost_prev,
        )
        return sqrt(dist)


    cdef bint support_unaligned(self) nogil:
        return True
