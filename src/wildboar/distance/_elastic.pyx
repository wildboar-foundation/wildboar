# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

# This implementation is heavily inspired by the UCRSuite.
#
# References
#
#  - Rakthanmanon, et al., Searching and Mining Trillions of Time
#    Series Subsequences under Dynamic Time Warping (2012)
#  - http://www.cs.ucr.edu/~eamonn/UCRsuite.html

cimport numpy as np

import numpy as np

from libc.math cimport INFINITY, exp, fabs, floor, sqrt
from libc.stdlib cimport free, labs, malloc
from libc.string cimport memcpy

from ..utils cimport stats
from ..utils.data cimport Dataset
from ..utils.misc cimport realloc_array
from ..utils.stats cimport fast_mean_std
from ._distance cimport (
    DistanceMeasure,
    ScaledSubsequenceDistanceMeasure,
    Subsequence,
    SubsequenceDistanceMeasure,
    SubsequenceView,
)

from sklearn.utils.validation import check_scalar


cdef struct DtwExtra:
    double *lower
    double *upper

cdef struct Deque:
    Py_ssize_t *queue
    int size
    int capacity
    int front
    int back


cdef void deque_init(Deque *c, Py_ssize_t capacity) nogil:
    c.capacity = capacity
    c.size = 0
    c.queue = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * capacity)
    c.front = 0
    c.back = capacity - 1


cdef void deque_reset(Deque *c) nogil:
    c.size = 0
    c.front = 0
    c.back = c.capacity - 1


cdef void deque_destroy(Deque *c) nogil:
    if c.queue != NULL:
        free(c.queue)
        c.queue = NULL


cdef void deque_push_back(Deque *c, Py_ssize_t v) nogil:
    c.queue[c.back] = v
    c.back -= 1
    if c.back < 0:
        c.back = c.capacity - 1

    c.size += 1


cdef void deque_pop_front(Deque *c) nogil:
    c.front -= 1
    if c.front < 0:
        c.front = c.capacity - 1
    c.size -= 1


cdef void deque_pop_back(Deque *c) nogil:
    c.back = (c.back + 1) % c.capacity
    c.size -= 1


cdef Py_ssize_t deque_front(Deque *c) nogil:
    cdef int tmp = c.front - 1
    if tmp < 0:
        tmp = c.capacity - 1
    return c.queue[tmp]


cdef Py_ssize_t deque_back(Deque *c) nogil:
    cdef int tmp = (c.back + 1) % c.capacity
    return c.queue[tmp]


cdef bint deque_empty(Deque *c) nogil:
    return c.size == 0


cdef Py_ssize_t deque_size(Deque *c) nogil:
    return c.size


cdef void find_min_max(
    double *T,
    Py_ssize_t length,
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
            upper[k] = T[deque_front(du)]
            lower[k] = T[deque_front(dl)]

        current = T[i]
        prev = T[(i - 1)]
        if current > prev:
            deque_pop_back(du)
            while (not deque_empty(du) and current > T[deque_back(du)]):
                deque_pop_back(du)
        else:
            deque_pop_back(dl)
            while (not deque_empty(dl) and current < T[deque_back(dl)]):
                deque_pop_back(dl)

        deque_push_back(du, i)
        deque_push_back(dl, i)

        if i == 2 * r + 1 + deque_front(du):
            deque_pop_front(du)
        elif i == 2 * r + 1 + deque_front(dl):
            deque_pop_front(dl)

    for i in range(length, length + r + 1):
        upper[i - r - 1] = T[deque_front(du)]
        lower[i - r - 1] = T[deque_front(dl)]

        if i - deque_front(du) >= 2 * r + 1:
            deque_pop_front(du)
        if i - deque_front(dl) >= 2 * r + 1:
            deque_pop_front(dl)


cdef inline double dist(double x, double y) nogil:
    cdef double s = x - y
    return s * s


cdef double constant_lower_bound(
    double *S,
    double s_mean,
    double s_std,
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
    t_x0 = (T[0] - t_mean) / t_std
    t_y0 = (T[length - 1] - t_mean) / t_std

    # first and last in S
    s_x0 = (S[0] - s_mean) / s_std
    s_y0 = (S[length - 1] - s_mean) / s_std

    min_dist = dist(t_x0, s_x0) + dist(t_y0, s_y0)
    if min_dist >= best_dist:
        return min_dist

    t_x1 = (T[1] - t_mean) / t_std
    s_x1 = (S[1] - s_mean) / s_std
    min_dist += min(
        min(dist(t_x1, s_x0), dist(t_x0, s_x1)),
        dist(t_x1, s_x1))

    if min_dist >= best_dist:
        return min_dist

    t_y1 = (T[length - 2] - t_mean) / t_std
    s_y1 = (S[length - 2] - s_mean) / s_std
    min_dist += min(
        min(dist(t_y1, s_y1), dist(t_y0, s_y1)),
        dist(t_y1, s_y1))

    if min_dist >= best_dist:
        return min_dist

    t_x2 = (T[2] - t_mean) / t_std
    s_x2 = (S[2] - s_mean) / s_std
    min_dist += min(min(dist(t_x0, s_x2),
                        min(dist(t_x1, s_x2),
                            dist(t_x2, s_x2)),
                        dist(t_x2, s_x1)),
                    dist(t_x2, s_x0))

    if min_dist >= best_dist:
        return min_dist

    t_y2 = (T[length - 3] - t_mean) / t_std
    s_y2 = (S[length - 3] - s_mean) / s_std

    min_dist += min(min(dist(t_y0, s_y2),
                        min(dist(t_y1, s_y2),
                            dist(t_y2, s_y2)),
                        dist(t_y2, s_y1)),
                    dist(t_y2, s_y0))

    return min_dist


cdef double cumulative_bound(
    double *T,
    Py_ssize_t length,
    double mean,
    double std,
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

        x = (T[i] - mean) / std
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
    double *S,
    int s_length,
    double s_mean,
    double s_std,
    double *X_buffer,
    double mean,
    double std,
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
                    v = (S[0] - s_mean) / s_std
                else:
                    v = 0

                if not std_zero:
                    v -= (X_buffer[0] - mean) / std
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

                v = (S[i] - s_mean) / s_std
                if not std_zero:
                    v -= (X_buffer[j] - mean) / std

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
    double *S,
    Py_ssize_t s_length,
    double s_mean,
    double s_std,
    double *T,
    Py_ssize_t t_length,
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
        current_value = T[i]
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
            lb_kim = constant_lower_bound(
                S,
                s_mean, 
                s_std, 
                X_buffer + j,
                mean, 
                std, 
                s_length, 
                min_dist,
            )

            if lb_kim < min_dist:
                lb_k = cumulative_bound(
                    X_buffer + j,
                    s_length, 
                    mean, 
                    std, 
                    s_mean, 
                    s_std, 
                    s_lower,
                    s_upper,
                    cb_1, 
                    min_dist,
                )
                if lb_k < min_dist:
                    lb_k2 = cumulative_bound(
                        S,
                        s_length, 
                        s_mean, 
                        s_std, 
                        mean, 
                        std, 
                        t_lower + I, 
                        t_upper + I, 
                        cb_2, 
                        min_dist,
                    )

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
                            S, 
                            s_length, 
                            s_mean,
                            s_std, 
                            X_buffer + j, 
                            mean, 
                            std, 
                            r, 
                            cb,
                            cost, 
                            cost_prev,
                            min_dist,
                        )

                        if dist < min_dist:
                            if index != NULL:
                                index[0] = (i + 1) - s_length
                            min_dist = dist

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return sqrt(min_dist)


cdef Py_ssize_t scaled_dtw_matches(
    double *S,
    Py_ssize_t s_length,
    double s_mean,
    double s_std,
    double *T,
    Py_ssize_t t_length,
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
    double threshold,
    double **distances,
    Py_ssize_t **matches,
) nogil:
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0

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

    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    threshold = threshold * threshold
    cdef Py_ssize_t n_matches = 0

    matches[0] = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * capacity)
    distances[0] = <double*> malloc(sizeof(double) * capacity)

    for i in range(t_length):
        current_value = T[i]
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
            lb_kim = constant_lower_bound(
                S,
                s_mean, 
                s_std, 
                X_buffer + j,
                mean, 
                std, 
                s_length, 
                threshold,
            )

            if lb_kim < threshold:
                lb_k = cumulative_bound(
                    X_buffer + j,
                    s_length, 
                    mean, 
                    std, 
                    s_mean, 
                    s_std, 
                    s_lower,
                    s_upper,
                    cb_1, 
                    threshold,
                )
                if lb_k < threshold:
                    lb_k2 = cumulative_bound(
                        S,
                        s_length, 
                        s_mean, 
                        s_std, 
                        mean, 
                        std, 
                        t_lower + I, 
                        t_upper + I, 
                        cb_2, 
                        threshold,
                    )

                    if lb_k2 < threshold:
                        if lb_k > lb_k2:
                            cb[s_length - 1] = cb_1[s_length - 1]
                            for k in range(s_length - 2, -1, -1):
                                cb[k] = cb[k + 1] + cb_1[k]
                        else:
                            cb[s_length - 1] = cb_2[s_length - 1]
                            for k in range(s_length - 2, -1, -1):
                                cb[k] = cb[k + 1] + cb_2[k]
                        dist = inner_scaled_dtw(
                            S, 
                            s_length, 
                            s_mean,
                            s_std, 
                            X_buffer + j, 
                            mean, 
                            std, 
                            r, 
                            cb,
                            cost, 
                            cost_prev,
                            threshold,
                        )

                        if dist <= threshold:
                            tmp_capacity = capacity
                            realloc_array(<void**> matches, n_matches, sizeof(Py_ssize_t), &tmp_capacity)
                            realloc_array(<void**> distances, n_matches, sizeof(double), &capacity)
                            matches[0][n_matches] = (i + 1) - s_length
                            distances[0][n_matches] = sqrt(dist)
                            n_matches += 1

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return n_matches


cdef double inner_dtw_subsequence(
    double *S,
    int s_length,
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
                v = T[0] - S[0]
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

                v = T[i] - S[j]
                cost[k] = min(min(x, y), z) + v * v

            k += 1
        cost_tmp = cost
        cost = cost_prev
        cost_prev = cost_tmp
    return cost_prev[k - 1]


cdef double dtw_subsequence_distance(
    double *S,
    Py_ssize_t s_length,
    double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    Py_ssize_t *index,
) nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef Py_ssize_t i
    for i in range(t_length - s_length + 1):
        dist = inner_dtw_subsequence(
            S,
            s_length, 
            T + i,
            r, 
            cost, 
            cost_prev, 
            min_dist,
        )
        if dist < min_dist:
            if index != NULL:
                index[0] = i
            min_dist = dist

    return sqrt(min_dist)


cdef Py_ssize_t dtw_subsequence_matches(
    double *S,
    Py_ssize_t s_length,
    double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double threshold,
    double **distances,
    Py_ssize_t **matches
) nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    threshold = threshold * threshold
    cdef Py_ssize_t i
    cdef Py_ssize_t n_matches = 0

    matches[0] = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * capacity)
    distances[0] = <double*> malloc(sizeof(double) * capacity)

    for i in range(t_length - s_length + 1):
        dist = inner_dtw_subsequence(
            S,
            s_length, 
            T + i,
            r, 
            cost, 
            cost_prev, 
            threshold,
        )

        if dist <= threshold:
            tmp_capacity = capacity
            realloc_array(<void**> matches, n_matches, sizeof(Py_ssize_t), &tmp_capacity)
            realloc_array(<void**> distances, n_matches, sizeof(double), &capacity)
            matches[0][n_matches] = i
            distances[0][n_matches] = sqrt(dist)
            n_matches += 1

    return n_matches


cdef double dtw_distance(
    double *X,
    Py_ssize_t x_length,
    double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double *weight_vector = NULL,
) nogil:
    """Dynamic time warping distance
    
    Parameters
    ----------
    X : data of x
    x_length : length of x
    x_mean : mean of array in x (if 0 ignored)
    x_std : std of array in x (or 1)
    Y : data of y
    y_length : length of y
    y_mean : mean of array in y (if 0 ignored)
    y_std : std of array in y (or 1)
    r : the warp window
    cost : cost matrix (max(x_length, y_length))
    cost_prev : cost matrix (max(x_length, y_length))
    weight_vector : weight vector (max(x_length, y_length))

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
    cdef double w = 1.0

    v = X[0] - Y[0]
    if weight_vector != NULL:
        w = weight_vector[0]
    
    cost_prev[0] = v * v * w
    for i in range(1, min(y_length, max(0, y_length - x_length) + r)):
        v = X[0] - Y[i]
        if weight_vector != NULL:
            w = weight_vector[i - 1]

        cost_prev[i] = cost_prev[i - 1] + v * v * w

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

            v = X[i] - Y[j]
            if weight_vector != NULL:
                w = weight_vector[labs(i - j)]

            cost[j] = min(min(x, y), z) + v * v * w

        if j_stop < y_length:
            cost[j_stop] = INFINITY

        cost, cost_prev = cost_prev, cost
    return cost_prev[y_length - 1]


cdef void _dtw_align(
    double[:] X, 
    double[:] Y, 
    Py_ssize_t r, 
    double[:] weights, 
    double[:,:] out
) nogil:
    """Compute the warp alignment """
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start, j_stop
    cdef double v
    cdef double x, y, z
    cdef double w = 1.0
    v = X[0] - Y[0]

    if weights is not None:
        w = weights[0]
    
    out[0, 0] = v * v * w
    for i in range(1, min(X.shape[0], r + 1)):
        v = X[i] - Y[0]
        if weights is not None:
            w = weights[i]

        out[i, 0] = out[i - 1, 0] + v * v * w

    for i in range(1, min(Y.shape[0], max(0, Y.shape[0] - X.shape[0]) + r)):
        v = X[0] - Y[i]
        if weights is not None:
            w = weights[i]

        out[0, i] = out[0, i - 1] + v * v * w

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
            if weights is not None:
                w = weights[labs(i - j)]

            out[i, j] = min(min(x, y), z) + v * v * w

        if j_stop < Y.shape[0]:
            out[i, j_stop] = INFINITY


def _dtw_alignment(
    np.ndarray x, 
    np.ndarray y, 
    Py_ssize_t r, 
    np.ndarray weights=None, 
    np.ndarray out=None
):
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
    if not 0 < r <= max(x.shape[0], y.shape[0]):
        raise ValueError("invalid r")
    cdef Py_ssize_t x_size = x.shape[0]
    cdef Py_ssize_t y_size = y.shape[0]
    if out is None:
        out = np.empty((x_size, y_size))

    if out.shape[0] < x_size or out.shape[1] < y_size:
        raise ValueError("out has wrong shape, got [%d, %d]" (x_size, y_size))
    _dtw_align(x, y, r, weights, out)
    return out


def _dtw_envelop(np.ndarray x, Py_ssize_t r):
    if not 0 < r <= x.shape[0]:
        raise ValueError("invalid r")

    x = np.ascontiguousarray(x, dtype=float)

    cdef Deque du
    cdef Deque dl
    cdef Py_ssize_t length = x.shape[0]
    cdef double *data = <double*> x.data
    cdef np.ndarray lower = np.empty(length, dtype=float)
    cdef np.ndarray upper = np.empty(length, dtype=float)
    cdef double *lower_data = <double*> lower.data
    cdef double *upper_data = <double*> upper.data

    deque_init(&dl, 2 * r + 2)
    deque_init(&du, 2 * r + 2)
    find_min_max(data, length, r, lower_data, upper_data, &dl, &du)

    deque_destroy(&dl)
    deque_destroy(&du)
    return lower, upper


def _dtw_lb_keogh(np.ndarray x, np.ndarray lower, np.ndarray upper, Py_ssize_t r):
    if not 0 < r <= x.shape[0]:
        raise ValueError("invalid r")
    x = np.ascontiguousarray(x, dtype=float)
    lower = np.ascontiguousarray(lower, dtype=float)
    upper = np.ascontiguousarray(upper, dtype=float)
    cdef Py_ssize_t i
    cdef Py_ssize_t length = x.shape[0]
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
    cdef np.ndarray cb = np.empty(length, dtype=float)
    cdef double *cb_data = <double*> cb.data
    cdef double min_dist
    min_dist = cumulative_bound(
        data, 
        length, 
        0, 
        1, 
        0, 
        1, 
        lower_data, 
        upper_data, 
        cb_data,
        INFINITY,
    )
    if <double*> upper.data != upper_data:
        free(upper_data)
    if <double*> lower.data != lower_data:
        free(lower_data)
    return sqrt(min_dist), cb


cdef double lcss_distance(
    double *X,
    Py_ssize_t x_length,
    double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double threshold,
    double *cost,
    double *cost_prev,
    double *weight_vector = NULL,
) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double v
    cdef double x, y, z
    cdef double w = 1.0
  
    for i in range(0, min(y_length, max(0, y_length - x_length) + r)):
        cost_prev[i] = 0

    if max(0, y_length - x_length) + r < y_length:
        cost_prev[max(0, y_length - x_length) + r] = 0

    for i in range(x_length):
        j_start = max(0, i - max(0, x_length - y_length) - r + 1)
        j_stop = min(y_length, i + max(0, y_length - x_length) + r)
        if j_start > 0:
            cost[j_start - 1] = 0

        for j in range(j_start, j_stop):
            x = cost_prev[j]
            if j > 0:
                y = cost_prev[j - 1]
                z = cost[j - 1]
            else:
                y = 0
                z = 0

            v = fabs(X[i] - Y[j])
            if weight_vector != NULL:
                w = weight_vector[labs(i - j)]

            if v <= threshold:
                cost[j] = w + y
            else:
                cost[j] = max(z, x)

        if j_stop < y_length:
            cost[j_stop] = 0

        cost, cost_prev = cost_prev, cost

    return 1 - (cost_prev[y_length - 1] / min(x_length, y_length))


cdef double erp_distance(
    double *X,
    Py_ssize_t x_length,
    double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double g,
    double *gX,
    double *gY,
    double *cost,
    double *cost_prev,
) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double v, x, y, z
    cdef double gx_sum = 0
    cdef double gy_sum = 0

    for i in range(x_length):
        v = fabs(X[i] - g)
        gX[i] = v
        gx_sum += v
    
    for i in range(y_length):
        v = fabs(Y[i] - g)
        gY[i] = v
        gy_sum += v

    for i in range(0, min(y_length, max(0, y_length - x_length) + r)):
        cost_prev[i] = gy_sum

    if max(0, y_length - x_length) + r < y_length:
        cost_prev[max(0, y_length - x_length) + r] = gy_sum

    for i in range(x_length):
        j_start = max(0, i - max(0, x_length - y_length) - r + 1)
        j_stop = min(y_length, i + max(0, y_length - x_length) + r)
        if j_start > 0:
            cost[j_start - 1] = 0

        for j in range(j_start, j_stop):
            x = cost_prev[j]
            if j > 0:
                y = cost_prev[j - 1]
                z = cost[j - 1]
            else:
                if i == 0:
                    # top-left is 0
                    y = 0 
                else:
                    # left column
                    y = gx_sum 

                z = gx_sum

            v = fabs(X[i] - Y[j])
            cost[j] = min(y + v, min(x + gX[i], z + gY[j]))

        if j_stop < y_length:
            cost[j_stop] = 0

        cost, cost_prev = cost_prev, cost

    return cost_prev[y_length - 1]


cdef double edr_distance(
    double *X,
    Py_ssize_t x_length,
    double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double threshold,
    double *cost,
    double *cost_prev,
    double *weight_vector = NULL,
) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double v
    cdef double x, y, z
    cdef double w = 1.0
  
    for i in range(0, min(y_length, max(0, y_length - x_length) + r)):
        cost_prev[i] = 0

    if max(0, y_length - x_length) + r < y_length:
        cost_prev[max(0, y_length - x_length) + r] = 0

    for i in range(x_length):
        j_start = max(0, i - max(0, x_length - y_length) - r + 1)
        j_stop = min(y_length, i + max(0, y_length - x_length) + r)
        if j_start > 0:
            cost[j_start - 1] = 0

        for j in range(j_start, j_stop):
            x = cost_prev[j]
            if j > 0:
                y = cost_prev[j - 1]
                z = cost[j - 1]
            else:
                y = 0
                z = 0

            v = fabs(X[i] - Y[j])
            if weight_vector != NULL:
                w = weight_vector[labs(i - j)]

            cost[j] = min(y + (0 if v < threshold else 1), x + 1, z + 1)

        if j_stop < y_length:
            cost[j_stop] = 0

        cost, cost_prev = cost_prev, cost

    return cost_prev[y_length - 1] / max(x_length, y_length)


cdef inline double _msm_cost(float x, float y, float z, float c) nogil:
    if y <= x <= z or y >= x >= z:
        return c
    else:
        return c + min(fabs(x - y), fabs(x - z))

# Stefan, A., Athitsos, V., & Das, G. (2013). 
#   The Move-Split-Merge Metric for Time Series. 
#   IEEE Transactions on Knowledge and Data Engineering, 25(6), 1425-1438.
cdef double msm_distance(
    double *X,
    Py_ssize_t x_length,
    double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double c,
    double *cost,
    double *cost_prev,
    double *cost_y,
) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop

    # Fill the first row
    cost_prev[0] = fabs(X[0] - Y[0])
    for i in range(1, min(y_length, max(0, y_length - x_length) + r)):
        cost_prev[i] = cost_prev[i - 1] + _msm_cost(Y[i], Y[i - 1], X[0], c)

    i = max(0, y_length - x_length) + r
    if i < y_length:
        cost_prev[i] = cost_prev[i - 1] + _msm_cost(Y[i], Y[i - 1], X[0], c)

    # Fill the first column
    cost_y[0] = cost_prev[0]
    for i in range(1, x_length):
        cost_y[i] = cost_y[i - 1] + _msm_cost(X[i], X[i - 1], Y[0], c)

    for i in range(1, x_length):
        j_start = max(1, i - max(0, x_length - y_length) - r + 1)
        j_stop = min(y_length, i + max(0, y_length - x_length) + r)

        cost[0] = cost_y[i]
        for j in range(j_start, j_stop):
            cost[j] = min(
                cost_prev[j - 1] + fabs(X[i] - Y[j]),
                cost_prev[j] + _msm_cost(X[i], X[i - 1], Y[j], c),    
                cost[j - 1] + _msm_cost(Y[j], X[i], Y[j - 1], c),
            )

        if j_stop < y_length:
            cost[j_stop] = 0

        cost, cost_prev = cost_prev, cost

    return cost_prev[y_length - 1]


cdef double twe_distance(
    double *X,
    Py_ssize_t x_length,
    double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double penalty,
    double stiffness,
    double *cost,
    double *cost_prev,
) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double v, x, y, del_x, del_y, match
    cdef double up_left, left, up

    for i in range(0, min(y_length, max(0, y_length - x_length) + r)):
        cost_prev[i] = INFINITY

    i = max(0, y_length - x_length) + r
    if i < y_length:
        cost_prev[i] = INFINITY

    penalty = penalty + stiffness
    for i in range(0, x_length):
        j_start = max(0, i - max(0, x_length - y_length) - r + 1)
        j_stop = min(y_length, i + max(0, y_length - x_length) + r)

        if j_start > 0:
            cost[j_start - 1] = 0

        for j in range(j_start, j_stop):
            up = cost_prev[j]
            if j == 0:
                left = INFINITY
                if i == 0:
                    up_left = 0
                else:
                    up_left = INFINITY
            else:
                left = cost[j - 1]
                up_left = cost_prev[j - 1]

            if i == 0:
                x = 0
                y = X[i]
            else:
                x = X[i - 1]
                y = X[i]
                
            del_x = up + fabs(x - y) + penalty

            if j == 0:
                x = 0
                y = Y[j]
            else:
                x = Y[j - 1]
                y = Y[j]

            del_y = left + fabs(x - y) + penalty
            
            if i == 0:
                x = 0
            else:
                x = X[i - 1]
            
            if j == 0:
                y = 0
            else:
                y = Y[j - 1]

            match = (
                up_left
                + fabs(X[i] - Y[j]) 
                + fabs(x - y)
                + stiffness * 2 * fabs(i - j)
            )

            cost[j] = min(del_x, del_y, match)

        if j_stop < y_length:
            cost[j_stop] = 0

        cost, cost_prev = cost_prev, cost

    return cost_prev[y_length - 1]


cdef Py_ssize_t _compute_warp_width(Py_ssize_t length, double r) nogil:
    if r == 1:
        return length - 1
    else:
        return <Py_ssize_t> floor(length * r)


cdef class ScaledDtwSubsequenceDistanceMeasure(ScaledSubsequenceDistanceMeasure):
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

    def __cinit__(self, double r=1.0):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        self.r = r
        self.X_buffer = NULL
        self.lower = NULL
        self.upper = NULL
        self.cost = NULL
        self.cost_prev = NULL
        self.cb = NULL
        self.cb_1 = NULL
        self.cb_2 = NULL
        self.dl.queue = NULL
        self.du.queue = NULL

    def __dealloc__(self):
        self._free()

    cdef void _free(self) nogil:
        if self.X_buffer != NULL:
            free(self.X_buffer)
            self.X_buffer = NULL
        if self.lower != NULL:            
            free(self.lower)
            self.lower = NULL
        if self.upper != NULL:
            free(self.upper)
            self.upper = NULL
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL
        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL
        if self.cb != NULL:
            free(self.cb)
            self.cb = NULL
        if self.cb_1 != NULL:
            free(self.cb_1)
            self.cb_1 = NULL
        if self.cb_2 != NULL:
            free(self.cb_2)
            self.cb_2 = NULL
        
        deque_destroy(&self.dl)
        deque_destroy(&self.du)

    def __reduce__(self):
        return self.__class__, (self.r, )

    cdef int reset(self, Dataset dataset) nogil:
        self._free()
        cdef Py_ssize_t n_timestep = dataset.n_timestep
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
            return -1
        deque_init(&self.dl, 2 * _compute_warp_width(n_timestep, self.r) + 2)
        deque_init(&self.du, 2 * _compute_warp_width(n_timestep, self.r) + 2)

    cdef double transient_distance(
        self,
        SubsequenceView *s,
        Dataset dataset,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)
        cdef DtwExtra *dtw_extra = <DtwExtra*> s.extra
        find_min_max(
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            warp_width,
            self.lower,
            self.upper,
            &self.dl,
            &self.du,
        )

        return scaled_dtw_distance(
            dataset.get_sample(s.index, s.dim) + s.start,
            s.length,
            s.mean,
            s.std if s.std != 0.0 else 1.0,
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
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
            return_index,
        )

    cdef double persistent_distance(
        self,
        Subsequence *s,
        Dataset dataset,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        cdef double *s_lower
        cdef double *s_upper
        cdef DtwExtra *extra
        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)

        if s.extra != NULL:
            extra = <DtwExtra*> s.extra
            s_lower = extra[0].lower
            s_upper = extra[0].upper
        else:
            with gil: print("********* ERROR ***********")
            s_lower = <double*> malloc(sizeof(double) * s.length)
            s_upper = <double*> malloc(sizeof(double) * s.length)

            find_min_max(
                s.data,
                s.length,
                warp_width,
                s_lower,
                s_upper,
                &self.dl,
                &self.du,
            )

        find_min_max(
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            warp_width,
            self.lower,
            self.upper,
            &self.dl,
            &self.du,
        )

        cdef double distance = scaled_dtw_distance(
            s.data,
            s.length,
            s.mean,
            s.std if s.std != 0.0 else 1.0,
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
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


    cdef Py_ssize_t transient_matches(
        self,
        SubsequenceView *s,
        Dataset dataset,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil:
        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)
        cdef DtwExtra *dtw_extra = <DtwExtra*> s.extra
        find_min_max(
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            warp_width,
            self.lower,
            self.upper,
            &self.dl,
            &self.du,
        )

        return scaled_dtw_matches(
            dataset.get_sample(s.index, s.dim) + s.start,
            s.length,
            s.mean,
            s.std if s.std != 0.0 else 1.0,
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
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
            threshold,
            distances,
            indicies,
        )

    cdef Py_ssize_t persistent_matches(
        self,
        Subsequence *s,
        Dataset dataset,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil:
        cdef double *s_lower
        cdef double *s_upper
        cdef DtwExtra *extra = <DtwExtra*> s.extra
        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)

        find_min_max(
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            warp_width,
            self.lower,
            self.upper,
            &self.dl,
            &self.du,
        )

        return scaled_dtw_matches(
            s.data,
            s.length,
            s.mean,
            s.std if s.std != 0.0 else 1.0,
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            warp_width,
            self.X_buffer,
            self.cost,
            self.cost_prev,
            extra.lower,
            extra.upper,
            self.lower,
            self.upper,
            self.cb,
            self.cb_1,
            self.cb_2,
            threshold,
            distances,
            indicies,
        )

    cdef int init_transient(
        self,
        Dataset dataset,
        SubsequenceView *t,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil:
        cdef int err = ScaledSubsequenceDistanceMeasure.init_transient(
            self, dataset, t, index, start, length, dim
        )
        if err < 0:
            return err

        cdef DtwExtra *dtw_extra = <DtwExtra*> malloc(sizeof(DtwExtra))
        dtw_extra.lower = <double*> malloc(sizeof(double) * length)
        dtw_extra.upper = <double*> malloc(sizeof(double) * length)
        cdef Py_ssize_t warp_width = _compute_warp_width(length, self.r)
        find_min_max(
            dataset.get_sample(index, dim) + start,
            length,
            warp_width,
            dtw_extra.lower,
            dtw_extra.upper,
            &self.dl,
            &self.du,
        )

        t.extra = dtw_extra
        return 0

    cdef int from_array(self, Subsequence *t, object obj):
        cdef int err = ScaledSubsequenceDistanceMeasure.from_array(self, t, obj)
        if err == -1:
            return -1
        dim, arr = obj
        cdef Py_ssize_t length = t.length
        cdef DtwExtra *dtw_extra = <DtwExtra*> malloc(sizeof(DtwExtra))
        dtw_extra.lower = <double*> malloc(sizeof(double) * length)
        dtw_extra.upper = <double*> malloc(sizeof(double) * length)

        cdef Py_ssize_t warp_width = _compute_warp_width(length, self.r)
        find_min_max(
            t.data,
            length,
            warp_width,
            dtw_extra.lower,
            dtw_extra.upper,
            &self.dl,
            &self.du,
        )
        t.extra = dtw_extra
        return 0

    cdef int init_persistent(
        self,
        Dataset dataset,
        SubsequenceView* v,
        Subsequence* s,
    ) nogil:
        cdef int err = ScaledSubsequenceDistanceMeasure.init_persistent(
            self, dataset, v, s
        )
        if err == -1:
            return -1

        cdef DtwExtra *dtw_extra = <DtwExtra*> malloc(sizeof(DtwExtra))
        dtw_extra.lower = <double*> malloc(sizeof(double) * v.length)
        dtw_extra.upper = <double*> malloc(sizeof(double) * v.length)

        cdef DtwExtra *orig = <DtwExtra*> v.extra
        memcpy(dtw_extra.lower, orig.lower, sizeof(double) * v.length)
        memcpy(dtw_extra.upper, orig.upper, sizeof(double) * v.length)
        s.extra = dtw_extra
        return 0

    cdef int free_transient(self, SubsequenceView *t) nogil:
        cdef DtwExtra *extra = <DtwExtra*> t.extra
        free(extra.lower)
        free(extra.upper)
        free(extra)
        return 0

    cdef int free_persistent(self, Subsequence *t) nogil:
        if t.data != NULL:
            free(t.data)
            t.data = NULL
        cdef DtwExtra *extra = <DtwExtra*> t.extra
        free(extra.lower)
        free(extra.upper)
        free(extra)
        return 0


cdef class DtwSubsequenceDistanceMeasure(SubsequenceDistanceMeasure):
    cdef double *cost
    cdef double *cost_prev
    cdef double r
    cdef Py_ssize_t cost_size


    def __cinit__(self, double r=1.0):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        self.r = r
        self.cost = NULL
        self.cost_prev = NULL
        self.cost_size = 0

    cdef int reset(self, Dataset dataset) nogil:
        self._free()
        cdef Py_ssize_t n_timestep = dataset.n_timestep
        self.cost_size = _compute_warp_width(n_timestep, self.r) * 2 + 1
        self.cost = <double*> malloc(sizeof(double) * self.cost_size)
        self.cost_prev = <double*> malloc(sizeof(double) * self.cost_size)

        if self.cost == NULL or self.cost_prev == NULL:
            return -1

    def __dealloc__(self):
        self._free()

    cdef void _free(self) nogil:
        if self.cost != NULL:
            free(self.cost)

        if self.cost_prev != NULL:
            free(self.cost_prev)

    def __reduce__(self):
        return self.__class__, (self.r, )

    cdef double persistent_distance(
        self,
        Subsequence *s,
        Dataset dataset,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)

        return dtw_subsequence_distance(
            s.data,
            s.length,
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            warp_width,
            self.cost,
            self.cost_prev,
            return_index,
        )

    cdef double transient_distance(
        self,
        SubsequenceView *s,
        Dataset dataset,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)
        return dtw_subsequence_distance(
            dataset.get_sample(s.index, s.dim) + s.start,
            s.length,
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            warp_width,
            self.cost,
            self.cost_prev,
            return_index,
        )

    cdef Py_ssize_t transient_matches(
        self,
        SubsequenceView *v,
        Dataset dataset,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil:
        cdef Py_ssize_t warp_width = _compute_warp_width(v.length, self.r)
        return dtw_subsequence_matches(
            dataset.get_sample(v.index, v.dim) + v.start,
            v.length,
            dataset.get_sample(index, v.dim),
            dataset.n_timestep,
            warp_width,
            self.cost,
            self.cost_prev,
            threshold,
            distances,
            indicies,
        )

    cdef Py_ssize_t persistent_matches(
        self,
        Subsequence *s,
        Dataset dataset,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil:
        cdef Py_ssize_t warp_width = _compute_warp_width(s.length, self.r)
        return dtw_subsequence_matches(
            s.data,
            s.length,
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            warp_width,
            self.cost,
            self.cost_prev,
            threshold,
            distances,
            indicies,
        )


cdef class DtwDistanceMeasure(DistanceMeasure):

    cdef double *cost
    cdef double *cost_prev
    cdef Py_ssize_t warp_width
    cdef double r
    
    def __cinit__(self, double r=1.0, *args, **kwargs):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        self.r = r
        self.cost = NULL
        self.cost_prev = NULL

    def __reduce__(self):
        return self.__class__, (self.r, )

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

    cdef int reset(self, Dataset x, Dataset y) nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(x.n_timestep, y.n_timestep)
        self.warp_width = <Py_ssize_t> max(floor(n_timestep * self.r), 1)
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len
    ) nogil:
        cdef double dist = dtw_distance(
            x,
            x_len,
            y,
            y_len,
            self.warp_width,
            self.cost,
            self.cost_prev,
        )

        return sqrt(dist)

    @property
    def is_elastic(self):
        return True


cdef void average_slope(double *q, Py_ssize_t len, double *d) nogil:
    cdef Py_ssize_t i, j
    j = 0
    for i in range(1, len - 1):
        d[j] = ((q[i] - q[i - 1]) + ((q[i + 1] - q[i - 1]) / 2)) / 2
        j += 1


cdef class DerivativeDtwDistanceMeasure(DtwDistanceMeasure):

    cdef double *d_x
    cdef double *d_y

    def __cinit__(self, double r=1.0):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        self.d_x = NULL
        self.d_y = NULL

    cdef void __free(self) nogil:
        DtwDistanceMeasure.__free(self)
        if self.d_x != NULL:
            free(self.d_x)
            self.d_x = NULL

        if self.d_y != NULL:
            free(self.d_y)
            self.d_y = NULL

    cdef int reset(self, Dataset x, Dataset y) nogil:
        DtwDistanceMeasure.reset(self, x, y)
        self.d_x = <double*> malloc(sizeof(double) * x.n_timestep - 2)
        self.d_y = <double*> malloc(sizeof(double) * y.n_timestep - 2)

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len,
    ) nogil:
        average_slope(x, x_len, self.d_x)
        average_slope(y, y_len, self.d_y)
        cdef double dist = dtw_distance(
            self.d_x,
            x_len - 2,
            self.d_y,
            y_len - 2,
            self.warp_width,
            self.cost,
            self.cost_prev,
        )

        return sqrt(dist)


cdef class WeightedDtwDistanceMeasure(DtwDistanceMeasure):

    cdef double g
    cdef double *weights

    def __cinit__(self, double r=1.0, double g=0.05):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        self.weights = NULL
        self.g = g

    def __reduce__(self):
        return self.__class__, (self.r, self.g)

    cdef void __free(self) nogil:
        DtwDistanceMeasure.__free(self)
        if self.weights != NULL:
            free(self.weights)
            self.weights = NULL

    cdef int reset(self, Dataset x, Dataset y) nogil:
        DtwDistanceMeasure.reset(self, x, y)
        cdef Py_ssize_t i
        cdef Py_ssize_t n_timestep = max(x.n_timestep, y.n_timestep)

        self.weights = <double*> malloc(sizeof(double) * n_timestep)
        for i in range(n_timestep):
            self.weights[i] = 1.0 / (1.0 + exp(-self.g * (i - n_timestep / 2.0)))

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len,
    ) nogil:
        cdef double dist = dtw_distance(
            x,
            x_len,
            y,
            y_len,
            self.warp_width,
            self.cost,
            self.cost_prev,
            self.weights,
        )

        return sqrt(dist)


cdef class WeightedDerivativeDtwDistanceMeasure(DtwDistanceMeasure):

    cdef double *d_x
    cdef double *d_y
    cdef double *weights
    cdef double g

    def __cinit__(self, double r=1.0, double g=0.05):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        self.weights = NULL
        self.g = g
        self.d_x = NULL
        self.d_y = NULL
    
    def __reduce__(self):
        return self.__class__, (self.r, self.g)

    cdef void __free(self) nogil:
        DtwDistanceMeasure.__free(self)
        if self.d_x != NULL:
            free(self.d_x)
            self.d_x = NULL

        if self.d_y != NULL:
            free(self.d_y)
            self.d_y = NULL

        if self.weights != NULL:
            free(self.weights)
            self.weights = NULL

    cdef int reset(self, Dataset x, Dataset y) nogil:
        DtwDistanceMeasure.reset(self, x, y)
        cdef Py_ssize_t i
        cdef Py_ssize_t n_timestep = max(x.n_timestep - 2, y.n_timestep - 2)

        self.d_x = <double*> malloc(sizeof(double) * x.n_timestep - 2)
        self.d_y = <double*> malloc(sizeof(double) * y.n_timestep - 2)
        self.weights = <double*> malloc(sizeof(double) * n_timestep)
        for i in range(0, n_timestep):
            self.weights[i] = 1.0 / (1.0 + exp(-self.g * (i - n_timestep / 2.0)))

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len,
    ) nogil:
        average_slope(x, x_len, self.d_x)
        average_slope(y, y_len, self.d_y)
        cdef double dist = dtw_distance(
            self.d_x,
            x_len - 2,
            self.d_y,
            y_len - 2,
            self.warp_width,
            self.cost,
            self.cost_prev,
            self.weights,
        )

        return sqrt(dist)

cdef class LcssDistanceMeasure(DistanceMeasure):

    cdef double *cost
    cdef double *cost_prev
    cdef Py_ssize_t warp_width
    cdef double r
    cdef double threshold
    
    def __cinit__(self, double r=1.0, double threshold=1.0, *args, **kwargs):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(threshold, "threshold", float, min_val=0)
        self.r = r
        self.threshold = threshold# * threshold
        self.cost = NULL
        self.cost_prev = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.threshold)

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

    cdef int reset(self, Dataset x, Dataset y) nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(x.n_timestep, y.n_timestep)
        self.warp_width = <Py_ssize_t> max(floor(n_timestep * self.r), 1)
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len
    ) nogil:
        cdef double dist = lcss_distance(
            x,
            x_len,
            y,
            y_len,
            self.warp_width,
            self.threshold,
            self.cost,
            self.cost_prev,
        )

        return dist

    @property
    def is_elastic(self):
        return True


cdef class WeightedLcssDistanceMeasure(LcssDistanceMeasure):

    cdef double g
    cdef double *weights

    def __cinit__(self, double r=1.0, threshold=1.0, double g=0.05):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(threshold, "threshold", float, min_val=0.0)
        self.weights = NULL
        self.g = g
        self.threshold = threshold

    def __reduce__(self):
        return self.__class__, (self.r, self.g)

    cdef void __free(self) nogil:
        LcssDistanceMeasure.__free(self)
        if self.weights != NULL:
            free(self.weights)
            self.weights = NULL

    cdef int reset(self, Dataset x, Dataset y) nogil:
        LcssDistanceMeasure.reset(self, x, y)
        cdef Py_ssize_t i
        cdef Py_ssize_t n_timestep = max(x.n_timestep, y.n_timestep)

        self.weights = <double*> malloc(sizeof(double) * n_timestep)
        for i in range(n_timestep):
            self.weights[i] = 1.0 / (1.0 + exp(-self.g * (i - n_timestep / 2.0)))

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len,
    ) nogil:
        cdef double dist = lcss_distance(
            x,
            x_len,
            y,
            y_len,
            self.warp_width,
            self.threshold,
            self.cost,
            self.cost_prev,
            self.weights,
        )

        return dist


cdef class ErpDistanceMeasure(DistanceMeasure):

    cdef double *cost
    cdef double *cost_prev
    cdef double *gX
    cdef double *gY

    cdef Py_ssize_t warp_width
    cdef double r
    cdef double g
    
    def __cinit__(self, double r=1.0, double g=0.0, *args, **kwargs):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(g, "g", float, min_val=0)
        self.r = r
        self.g = g
        self.cost = NULL
        self.cost_prev = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.g)

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

        if self.gX != NULL:
            free(self.gX)
            self.gX = NULL

        if self.gY != NULL:
            free(self.gY)
            self.gY = NULL    

    cdef int reset(self, Dataset x, Dataset y) nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(x.n_timestep, y.n_timestep)
        self.warp_width = <Py_ssize_t> max(floor(n_timestep * self.r), 1)
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)
        self.gX = <double*> malloc(sizeof(double) * x.n_timestep)
        self.gY = <double*> malloc(sizeof(double) * y.n_timestep)

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len
    ) nogil:
        cdef double dist = erp_distance(
            x,
            x_len,
            y,
            y_len,
            self.warp_width,
            self.g,
            self.gX,
            self.gY,
            self.cost,
            self.cost_prev,
        )

        return dist

    @property
    def is_elastic(self):
        return True


cdef class EdrDistanceMeasure(DistanceMeasure):

    cdef double *cost
    cdef double *cost_prev
    cdef double *std_x
    cdef double *std_y

    cdef Py_ssize_t warp_width
    cdef double r
    cdef double threshold
    
    def __cinit__(self, double r=1.0, double threshold=-INFINITY, *args, **kwargs):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        if threshold != -INFINITY:
            check_scalar(threshold, "threshold", float, min_val=0)
        self.r = r
        self.threshold = threshold
        self.cost = NULL
        self.cost_prev = NULL
        self.std_x = NULL
        self.std_y = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.threshold)

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

        if self.std_x != NULL:
            free(self.std_x)
            self.std_x = NULL

        if self.std_y != NULL:
            free(self.std_y)
            self.std_y = NULL

    cdef int reset(self, Dataset x, Dataset y) nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(x.n_timestep, y.n_timestep)
        self.warp_width = <Py_ssize_t> max(floor(n_timestep * self.r), 1)
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)
        self.std_x = <double*> malloc(sizeof(double) * x.n_samples)
        self.std_y = <double*> malloc(sizeof(double) * x.n_samples)
        
        cdef double mean, std
        cdef Py_ssize_t i
        if self.threshold == -INFINITY:
            for i in range(x.n_samples):
                fast_mean_std(x.get_sample(i, 0), x.n_timestep, &mean, &std)
                self.std_x[i] = std

            for i in range(y.n_samples):
                fast_mean_std(y.get_sample(i, 0), y.n_timestep, &mean, &std)
                self.std_y[i] = std

    cdef double distance(
        self,
        Dataset x,
        Py_ssize_t x_index,
        Dataset y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) nogil:
        if self.threshold == -INFINITY:
            threshold = max(self.std_x[x_index], self.std_y[y_index]) / 4.0
        else:
            threshold = self.threshold

        return edr_distance(
            x.get_sample(x_index, dim),
            x.n_timestep,
            y.get_sample(y_index, dim),
            y.n_timestep,
            self.warp_width,
            threshold,
            self.cost,
            self.cost_prev,
        )


    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len,
    ) nogil:
        cdef double mean, std_x, std_y, threshold
        if self.threshold == -INFINITY:
            fast_mean_std(x, x_len, &mean, &std_x)
            fast_mean_std(y, y_len, &mean, &std_y)
            threshold = max(std_x, std_y) / 4.0
        else:
            threshold = self.threshold

        return edr_distance(
            x,
            x_len,
            y,
            y_len,
            self.warp_width,
            threshold,
            self.cost,
            self.cost_prev,
        )

    @property
    def is_elastic(self):
        return True


cdef class MsmDistanceMeasure(DistanceMeasure):

    cdef double *cost
    cdef double *cost_prev
    cdef double *cost_y

    cdef Py_ssize_t warp_width
    cdef double r
    cdef double c
    
    def __cinit__(self, double r=1.0, double c=1.0, *args, **kwargs):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(c, "c", float, min_val=0)
        self.r = r
        self.c = c
        self.cost = NULL
        self.cost_prev = NULL
        self.cost_y = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.c)

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

        if self.cost_y != NULL:
            free(self.cost_y)
            self.cost_y = NULL

    cdef int reset(self, Dataset x, Dataset y) nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(x.n_timestep, y.n_timestep)
        self.warp_width = <Py_ssize_t> max(floor(n_timestep * self.r), 1)
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_y = <double*> malloc(sizeof(double) * x.n_timestep)

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len
    ) nogil:
        return msm_distance(
            x,
            x_len,
            y,
            y_len,
            self.warp_width,
            self.c,
            self.cost,
            self.cost_prev,
            self.cost_y,
        )

    @property
    def is_elastic(self):
        return True


cdef class TweDistanceMeasure(DistanceMeasure):

    cdef double *cost
    cdef double *cost_prev
    cdef Py_ssize_t warp_width
    cdef double r
    cdef double penalty
    cdef double stiffness
    
    def __cinit__(
        self, 
        double r=1.0, 
        double penalty=1.0, 
        stiffness=0.001, 
        *args, 
        **kwargs
    ):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(
            penalty, 
            "penalty", 
            float, 
            min_val=0.0, 
            include_boundaries="neither"
        )
        check_scalar(stiffness, "stiffness", float, min_val=0.0)
        self.r = r
        self.penalty = penalty
        self.stiffness = stiffness
        self.cost = NULL
        self.cost_prev = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.penalty, self.stiffness)

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

    cdef int reset(self, Dataset x, Dataset y) nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(x.n_timestep, y.n_timestep)
        self.warp_width = <Py_ssize_t> max(floor(n_timestep * self.r), 1)
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len
    ) nogil:
        return twe_distance(
            x,
            x_len,
            y,
            y_len,
            self.warp_width,
            self.penalty,
            self.stiffness,
            self.cost,
            self.cost_prev,
        )

    @property
    def is_elastic(self):
        return True