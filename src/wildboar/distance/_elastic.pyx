# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

import warnings

import numpy as np

from libc.math cimport INFINITY, NAN, exp, fabs, floor, isnan, sqrt, isinf
from libc.stdlib cimport free, labs, malloc
from libc.string cimport memcpy

from ..utils cimport TSArray
from ..utils._misc cimport realloc_array
from ..utils._stats cimport fast_mean_std
from ._metric cimport euclidean_distance
from ._cdistance cimport (
    Metric,
    ScaledSubsequenceMetric,
    Subsequence,
    SubsequenceMetric,
    SubsequenceView,
)

from sklearn.utils.validation import check_scalar


cdef struct DtwExtra:
    double *lower
    double *upper

cdef void deque_init(Deque *c, Py_ssize_t capacity) noexcept nogil:
    c.capacity = capacity
    c.size = 0
    c.queue = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * capacity)
    c.front = 0
    c.back = capacity - 1


cdef void deque_reset(Deque *c) noexcept nogil:
    c.size = 0
    c.front = 0
    c.back = c.capacity - 1


cdef void deque_destroy(Deque *c) noexcept nogil:
    if c.queue != NULL:
        free(c.queue)
        c.queue = NULL


cdef void deque_push_back(Deque *c, Py_ssize_t v) noexcept nogil:
    c.queue[c.back] = v
    c.back -= 1
    if c.back < 0:
        c.back = c.capacity - 1

    c.size += 1


cdef void deque_pop_front(Deque *c) noexcept nogil:
    c.front -= 1
    if c.front < 0:
        c.front = c.capacity - 1
    c.size -= 1


cdef void deque_pop_back(Deque *c) noexcept nogil:
    c.back = (c.back + 1) % c.capacity
    c.size -= 1


cdef Py_ssize_t deque_front(Deque *c) noexcept nogil:
    cdef int tmp = c.front - 1
    if tmp < 0:
        tmp = c.capacity - 1
    return c.queue[tmp]


cdef Py_ssize_t deque_back(Deque *c) noexcept nogil:
    cdef int tmp = (c.back + 1) % c.capacity
    return c.queue[tmp]


cdef bint deque_empty(Deque *c) noexcept nogil:
    return c.size == 0


cdef Py_ssize_t deque_size(Deque *c) noexcept nogil:
    return c.size


cdef void find_min_max(
    const double *T,
    Py_ssize_t length,
    Py_ssize_t r,
    double *lower,
    double *upper,
    Deque *dl,
    Deque *du,
) noexcept nogil:
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


cdef inline double dist(double x, double y) noexcept nogil:
    cdef double s = x - y
    return s * s


cdef double constant_lower_bound(
    const double *S,
    double s_mean,
    double s_std,
    const double *T,
    double t_mean,
    double t_std,
    Py_ssize_t length,
    double best_dist,
) noexcept nogil:
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
    const double *T,
    Py_ssize_t length,
    double mean,
    double std,
    double lu_mean,
    double lu_std,
    double *lower,
    double *upper,
    double *cb,
    double best_so_far,
) noexcept nogil:
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


cdef inline double inner_scaled_dtw_subsequence_distance(
    const double *S,
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
) noexcept nogil:
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

# This implementation is heavily inspired by the UCRSuite.
#
# References
#
#  - Rakthanmanon, et al., Searching and Mining Trillions of Time
#    Series Subsequences under Dynamic Time Warping (2012)
#  - http://www.cs.ucr.edu/~eamonn/UCRsuite.html
cdef double scaled_dtw_subsequence_distance(
    const double *S,
    Py_ssize_t s_length,
    double s_mean,
    double s_std,
    const double *T,
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
) noexcept nogil:
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
                        dist = inner_scaled_dtw_subsequence_distance(
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
    const double *S,
    Py_ssize_t s_length,
    double s_mean,
    double s_std,
    const double *T,
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
    double *distances,
    Py_ssize_t *matches,
) noexcept nogil:
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

    threshold = threshold * threshold
    cdef Py_ssize_t n_matches = 0

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
                        dist = inner_scaled_dtw_subsequence_distance(
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
                            if matches != NULL:
                                matches[n_matches] = (i + 1) - s_length
                            distances[n_matches] = sqrt(dist)
                            n_matches += 1

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return n_matches


cdef double dtw_subsequence_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY
    cdef Py_ssize_t length = t_length - s_length + 1

    cdef Py_ssize_t i
    for i in range(length):
        dist = dtw_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            cost,
            cost_prev,
            weight_vector,
            min_dist,
        )
        if dist < min_dist:
            if index != NULL:
                index[0] = i
            min_dist = dist

    return sqrt(min_dist)


cdef Py_ssize_t dtw_subsequence_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double threshold,
    double *distances,
    Py_ssize_t *matches
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    threshold = threshold * threshold
    cdef Py_ssize_t i
    cdef Py_ssize_t n_matches = 0
    cdef Py_ssize_t length = t_length - s_length + 1

    for i in range(length):
        dist = dtw_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            cost,
            cost_prev,
            weight_vector,
            threshold,
        )

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = sqrt(dist)
            n_matches += 1

    return n_matches


cdef double adtw_subsequence_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double penalty,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY
    cdef Py_ssize_t length = t_length - s_length + 1

    cdef Py_ssize_t i
    for i in range(length):
        dist = adtw_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            cost,
            cost_prev,
            penalty,
            min_dist,
        )
        if dist < min_dist:
            if index != NULL:
                index[0] = i
            min_dist = dist

    return sqrt(min_dist)


cdef Py_ssize_t adtw_subsequence_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double penalty,
    double threshold,
    double *distances,
    Py_ssize_t *matches
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    threshold = threshold * threshold
    cdef Py_ssize_t i
    cdef Py_ssize_t n_matches = 0
    cdef Py_ssize_t length = t_length - s_length + 1

    for i in range(length):
        dist = adtw_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            cost,
            cost_prev,
            penalty,
            threshold,
        )

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = sqrt(dist)
            n_matches += 1

    return n_matches


cdef double ddtw_subsequence_distance(
    const double *S,  # Assumed to be derivative
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double *T_buffer,
    Py_ssize_t *index,
) noexcept nogil:
    if s_length < 3:
        return 0

    cdef double dist = 0
    cdef double min_dist = INFINITY
    cdef Py_ssize_t length = t_length - s_length + 1

    cdef Py_ssize_t i
    for i in range(length):
        average_slope(T + i, s_length, T_buffer)
        dist = dtw_distance(
            S,
            s_length - 2,
            T_buffer,
            s_length - 2,
            r,
            cost,
            cost_prev,
            weight_vector,
            min_dist,
        )
        if dist < min_dist:
            if index != NULL:
                index[0] = i
            min_dist = dist

    return sqrt(min_dist)


cdef Py_ssize_t ddtw_subsequence_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double *T_buffer,
    double threshold,
    double *distances,
    Py_ssize_t *matches
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    threshold = threshold * threshold
    cdef Py_ssize_t i
    cdef Py_ssize_t n_matches = 0
    cdef Py_ssize_t length = t_length - s_length + 1

    if s_length < 3:
        return 0

    for i in range(length):
        average_slope(T + i, s_length, T_buffer)
        dist = dtw_distance(
            S,
            s_length - 2,
            T_buffer,
            s_length - 2,
            r,
            cost,
            cost_prev,
            weight_vector,
            threshold,
        )

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = sqrt(dist)
            n_matches += 1

    return n_matches


cdef double dtw_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double min_dist,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double x
    cdef double y
    cdef double z
    cdef double v
    cdef double w = 1.0
    cdef double min_cost
    cdef Py_ssize_t max_len = max(0, y_length - x_length) + r
    cdef Py_ssize_t min_len = max(0, x_length - y_length)

    v = X[0] - Y[0]
    if weight_vector != NULL:
        w = weight_vector[0]

    cost_prev[0] = v * v * w
    for i in range(1, min(y_length, max_len)):
        v = X[0] - Y[i]
        if weight_vector != NULL:
            w = weight_vector[i - 1]

        cost_prev[i] = cost_prev[i - 1] + v * v * w

    if max_len < y_length:
        cost_prev[max_len] = INFINITY

    for i in range(1, x_length):
        j_start = max(0, i - min_len - r + 1)
        j_stop = min(y_length, i + max_len)
        if j_start > 0:
            cost[j_start - 1] = INFINITY

        min_cost = INFINITY
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

            if cost[j] < min_cost:
                min_cost = cost[j]

        if min_cost > min_dist:
            return INFINITY

        if j_stop < y_length:
            cost[j_stop] = INFINITY

        cost, cost_prev = cost_prev, cost
    return cost_prev[y_length - 1]


cdef double adtw_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double penalty,
    double min_dist,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double x
    cdef double y
    cdef double z
    cdef double v
    cdef double min_cost
    cdef Py_ssize_t max_len = max(0, y_length - x_length) + r
    cdef Py_ssize_t min_len = max(0, x_length - y_length)
    v = X[0] - Y[0]

    cost_prev[0] = v * v
    for i in range(1, min(y_length, max_len)):
        v = X[0] - Y[i]
        cost_prev[i] = cost_prev[i - 1] + v * v

    if max_len < y_length:
        cost_prev[max_len] = INFINITY

    for i in range(1, x_length):
        j_start = max(0, i - min_len - r + 1)
        j_stop = min(y_length, i + max_len)
        if j_start > 0:
            cost[j_start - 1] = INFINITY

        min_cost = INFINITY
        for j in range(j_start, j_stop):
            x = cost_prev[j] + penalty
            if j > 0:
                y = cost[j - 1] + penalty
                z = cost_prev[j - 1]
            else:
                y = INFINITY
                z = INFINITY

            v = X[i] - Y[j]
            cost[j] = min(min(x, y), z) + v * v

            if cost[j] < min_cost:
                min_cost = cost[j]

        if min_cost > min_dist:
            return INFINITY

        if j_stop < y_length:
            cost[j_stop] = INFINITY

        cost, cost_prev = cost_prev, cost
    return cost_prev[y_length - 1]


def _dtw_alignment(
    const double[:] X,
    const double[:] Y,
    Py_ssize_t r,
    const double[:] weights,
    double[:, :] out = None
):
    if not 0 < r <= max(X.shape[0], Y.shape[0]):
        raise ValueError("invalid r")

    if out is None:
        out = np.empty((X.shape[0], Y.shape[0]))
    elif out.shape[0] < X.shape[0] or out.shape[1] < Y.shape[0]:
        raise ValueError("out has wrong shape, got [%d, %d]" % out.shape)

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

    return out.base


def _dtw_envelop(const double[::1] x, Py_ssize_t r):
    if not 0 < r <= x.shape[0]:
        raise ValueError("invalid r")

    cdef Deque du
    cdef Deque dl
    cdef Py_ssize_t length = x.shape[0]
    cdef double[:] lower = np.empty(length, dtype=float)
    cdef double[:] upper = np.empty(length, dtype=float)

    deque_init(&dl, 2 * r + 2)
    deque_init(&du, 2 * r + 2)
    find_min_max(&x[0], length, r, &lower[0], &upper[0], &dl, &du)

    deque_destroy(&dl)
    deque_destroy(&du)
    return lower.base, upper.base


def _dtw_lb_keogh(
    const double[::1] x, double[::1] lower, double[::1] upper, Py_ssize_t r
):
    if not 0 < r <= x.shape[0]:
        raise ValueError("invalid r")

    cdef Py_ssize_t i
    cdef Py_ssize_t length = x.shape[0]
    cdef double[:] cb = np.empty(length, dtype=float)
    cdef double min_dist
    min_dist = cumulative_bound(
        &x[0],
        length,
        0,
        1,
        0,
        1,
        &lower[0],
        &upper[0],
        &cb[0],
        INFINITY,
    )

    return sqrt(min_dist), cb.base


cdef double lcss_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double epsilon,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double min_dist,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double v
    cdef double x, y, z
    cdef double w = 1.0
    cdef double min_cost
    cdef Py_ssize_t max_len = max(0, y_length - x_length) + r
    cdef Py_ssize_t min_len = max(0, x_length - y_length)

    for i in range(0, min(y_length, max_len)):
        cost_prev[i] = 0

    if max_len < y_length:
        cost_prev[max_len] = 0

    for i in range(x_length):
        j_start = max(0, i - min_len - r + 1)
        j_stop = min(y_length, i + max_len)
        if j_start > 0:
            cost[j_start - 1] = 0

        min_cost = INFINITY
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

            if v <= epsilon:
                cost[j] = w + y
            else:
                cost[j] = max(z, x)

            if cost[j] < min_cost:
                min_cost = cost[j]

        if min_cost > min_dist:
            return INFINITY

        if j_stop < y_length:
            cost[j_stop] = 0

        cost, cost_prev = cost_prev, cost

    return 1 - (cost_prev[y_length - 1] / min(x_length, y_length))


cdef double lcss_subsequence_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double epsilon,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY
    cdef Py_ssize_t length = t_length - s_length + 1

    cdef Py_ssize_t i
    for i in range(length):
        dist = lcss_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            epsilon,
            cost,
            cost_prev,
            weight_vector,
            min(s_length, t_length) - min_dist * min(s_length, t_length)
            if isinf(min_dist) == 0
            else min_dist,
        )

        if dist < min_dist:
            if index != NULL:
                index[0] = i
            min_dist = dist

    return min_dist


cdef Py_ssize_t lcss_subsequence_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double epsilon,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double threshold,
    double *distances,
    Py_ssize_t *matches
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t n_matches = 0
    cdef Py_ssize_t length = t_length - s_length + 1

    for i in range(length):
        dist = lcss_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            epsilon,
            cost,
            cost_prev,
            weight_vector,
            min(s_length, t_length) - threshold * min(s_length, t_length)
            if isinf(threshold) == 0  # Ensure that passing inf as threshold works.
            else threshold,
        )

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = dist
            n_matches += 1

    return n_matches


cdef double erp_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double g,
    double *gX,
    double *gY,
    double *cost,
    double *cost_prev,
    double min_dist,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double v, x, y, z
    cdef double gx_sum = 0
    cdef double gy_sum = 0
    cdef double min_cost

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

        min_cost = INFINITY
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

            if cost[j] < min_cost:
                min_cost = cost[j]

        if min_cost > min_dist:
            return INFINITY

        if j_stop < y_length:
            cost[j_stop] = 0

        cost, cost_prev = cost_prev, cost

    return cost_prev[y_length - 1]


cdef double erp_subsequence_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double g,
    double *gX,
    double *gY,
    double *cost,
    double *cost_prev,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY
    cdef Py_ssize_t length = t_length - s_length + 1

    cdef Py_ssize_t i
    for i in range(length):
        dist = erp_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            g,
            gX,
            gY,
            cost,
            cost_prev,
            min_dist,
        )

        if dist < min_dist:
            if index != NULL:
                index[0] = i
            min_dist = dist

    return min_dist


cdef Py_ssize_t erp_subsequence_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double g,
    double *gX,
    double *gY,
    double *cost,
    double *cost_prev,
    double threshold,
    double *distances,
    Py_ssize_t *matches
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t n_matches = 0
    cdef Py_ssize_t length = t_length - s_length + 1

    for i in range(length):
        dist = erp_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            g,
            gX,
            gY,
            cost,
            cost_prev,
            threshold,
        )

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = dist
            n_matches += 1

    return n_matches


cdef double edr_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double epsilon,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double min_dist,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double v
    cdef double x, y, z
    cdef double w = 1.0
    cdef double min_cost

    for i in range(0, min(y_length, max(0, y_length - x_length) + r)):
        cost_prev[i] = 0

    if max(0, y_length - x_length) + r < y_length:
        cost_prev[max(0, y_length - x_length) + r] = 0

    for i in range(x_length):
        j_start = max(0, i - max(0, x_length - y_length) - r + 1)
        j_stop = min(y_length, i + max(0, y_length - x_length) + r)
        if j_start > 0:
            cost[j_start - 1] = 0

        min_cost = INFINITY
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

            cost[j] = min(y + (0 if v < epsilon else 1), x + 1, z + 1)

            if cost[j] < min_cost:
                min_cost = cost[j]

        if min_cost > min_dist:
            return INFINITY

        if j_stop < y_length:
            cost[j_stop] = 0

        cost, cost_prev = cost_prev, cost

    return cost_prev[y_length - 1] / max(x_length, y_length)


cdef double edr_subsequence_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double epsilon,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY
    cdef Py_ssize_t length = t_length - s_length + 1

    cdef Py_ssize_t i
    for i in range(length):
        dist = edr_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            epsilon,
            cost,
            cost_prev,
            weight_vector,
            min_dist * max(s_length, t_length),
        )

        if dist < min_dist:
            if index != NULL:
                index[0] = i
            min_dist = dist

    return min_dist


cdef Py_ssize_t edr_subsequence_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double epsilon,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double threshold,
    double *distances,
    Py_ssize_t *matches
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t n_matches = 0
    cdef Py_ssize_t length = t_length - s_length + 1

    for i in range(length):
        dist = edr_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            epsilon,
            cost,
            cost_prev,
            weight_vector,
            threshold * max(s_length, t_length),
        )

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = dist
            n_matches += 1

    return n_matches


cdef inline double _msm_cost(float x, float y, float z, float c) noexcept nogil:
    if y <= x <= z or y >= x >= z:
        return c
    else:
        return c + min(fabs(x - y), fabs(x - z))

# Stefan, A., Athitsos, V., & Das, G. (2013).
#   The Move-Split-Merge Metric for Time Series.
#   IEEE Transactions on Knowledge and Data Engineering, 25(6), 1425-1438.
cdef double msm_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double c,
    double *cost,
    double *cost_prev,
    double *cost_y,
    double min_dist,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double min_cost

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
        min_cost = cost[0]
        for j in range(j_start, j_stop):
            cost[j] = min(
                cost_prev[j - 1] + fabs(X[i] - Y[j]),
                cost_prev[j] + _msm_cost(X[i], X[i - 1], Y[j], c),
                cost[j - 1] + _msm_cost(Y[j], X[i], Y[j - 1], c),
            )
            if cost[j] < min_cost:
                min_cost = cost[j]

        if min_cost > min_dist:
            return INFINITY

        if j_stop < y_length:
            cost[j_stop] = 0

        cost, cost_prev = cost_prev, cost

    return cost_prev[y_length - 1]


cdef double msm_subsequence_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double c,
    double *cost,
    double *cost_prev,
    double *cost_y,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY
    cdef Py_ssize_t length = t_length - s_length + 1

    cdef Py_ssize_t i
    for i in range(length):
        dist = msm_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            c,
            cost,
            cost_prev,
            cost_y,
            min_dist,
        )

        if dist < min_dist:
            if index != NULL:
                index[0] = i
            min_dist = dist

    return min_dist


cdef Py_ssize_t msm_subsequence_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double c,
    double *cost,
    double *cost_prev,
    double *cost_y,
    double threshold,
    double *distances,
    Py_ssize_t *matches
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t n_matches = 0
    cdef Py_ssize_t length = t_length - s_length + 1

    for i in range(length):
        dist = msm_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            c,
            cost,
            cost_prev,
            cost_y,
            threshold,
        )

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = dist
            n_matches += 1

    return n_matches


cdef double twe_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double penalty,
    double stiffness,
    double *cost,
    double *cost_prev,
    double min_dist,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t j_start
    cdef Py_ssize_t j_stop
    cdef double v, x, y, del_x, del_y, match
    cdef double up_left, left, up
    cdef double min_cost

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

        min_cost = INFINITY
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
                + stiffness * 2 * labs(i - j)
            )

            cost[j] = min(del_x, del_y, match)

            if cost[j] < min_cost:
                min_cost = cost[j]

        if min_cost > min_dist:
            return INFINITY

        if j_stop < y_length:
            cost[j_stop] = 0

        cost, cost_prev = cost_prev, cost

    return cost_prev[y_length - 1]


cdef double twe_subsequence_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double penalty,
    double stiffness,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY
    cdef Py_ssize_t length = t_length - s_length + 1

    cdef Py_ssize_t i
    for i in range(length):
        dist = twe_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            penalty,
            stiffness,
            cost,
            cost_prev,
            min_dist,
        )

        if dist < min_dist:
            if index != NULL:
                index[0] = i
            min_dist = dist

    return min_dist


cdef Py_ssize_t twe_subsequence_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double penalty,
    double stiffness,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double threshold,
    double *distances,
    Py_ssize_t *matches
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t n_matches = 0
    cdef Py_ssize_t length = t_length - s_length + 1

    for i in range(length):
        dist = twe_distance(
            S,
            s_length,
            T + i,
            s_length,
            r,
            penalty,
            stiffness,
            cost,
            cost_prev,
            threshold,
        )

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = dist
            n_matches += 1

    return n_matches


cdef Py_ssize_t _compute_warp_width(Py_ssize_t length, double r) noexcept nogil:
    if r == 1:
        return length - 1
    else:
        return <Py_ssize_t> floor(length * r)


cdef inline Py_ssize_t _compute_r(Py_ssize_t length, double r) noexcept nogil:
    return <Py_ssize_t> max(floor(length * r), 1)


cdef class ScaledDtwSubsequenceMetric(ScaledSubsequenceMetric):
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

    cdef void _free(self) noexcept nogil:
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

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        cdef Py_ssize_t n_timestep = X.shape[2]
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

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        cdef Py_ssize_t warp_width = _compute_warp_width(s_len, self.r)
        cdef DtwExtra *dtw_extra = <DtwExtra*> s_extra
        find_min_max(
            x,
            x_len,
            warp_width,
            self.lower,
            self.upper,
            &self.dl,
            &self.du,
        )

        return scaled_dtw_subsequence_distance(
            s,
            s_len,
            s_mean,
            s_std,
            x,
            x_len,
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

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        cdef Py_ssize_t warp_width = _compute_warp_width(s_len, self.r)
        cdef DtwExtra *dtw_extra = <DtwExtra*> s_extra
        find_min_max(
            x,
            x_len,
            warp_width,
            self.lower,
            self.upper,
            &self.dl,
            &self.du,
        )

        return scaled_dtw_matches(
            s,
            s_len,
            s_mean,
            s_std,
            x,
            x_len,
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

    cdef int init_transient(
        self,
        TSArray X,
        SubsequenceView *t,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) noexcept nogil:
        cdef int err = ScaledSubsequenceMetric.init_transient(
            self, X, t, index, start, length, dim
        )
        if err < 0:
            return err

        cdef DtwExtra *dtw_extra = <DtwExtra*> malloc(sizeof(DtwExtra))
        dtw_extra.lower = <double*> malloc(sizeof(double) * length)
        dtw_extra.upper = <double*> malloc(sizeof(double) * length)
        cdef Py_ssize_t warp_width = _compute_warp_width(length, self.r)
        find_min_max(
            &X[index, dim, start],
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
        cdef int err = ScaledSubsequenceMetric.from_array(self, t, obj)
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
        TSArray X,
        SubsequenceView* v,
        Subsequence* s,
    ) noexcept nogil:
        cdef int err = ScaledSubsequenceMetric.init_persistent(self, X, v, s)
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

    cdef int free_transient(self, SubsequenceView *t) noexcept nogil:
        cdef DtwExtra *extra = <DtwExtra*> t.extra
        free(extra.lower)
        free(extra.upper)
        free(extra)
        return 0

    cdef int free_persistent(self, Subsequence *t) noexcept nogil:
        if t.data != NULL:
            free(t.data)
            t.data = NULL
        cdef DtwExtra *extra = <DtwExtra*> t.extra
        free(extra.lower)
        free(extra.upper)
        free(extra)
        return 0


cdef class DtwSubsequenceMetric(SubsequenceMetric):
    cdef double *cost
    cdef double *cost_prev
    cdef double r

    def __init__(self, double r=1.0):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        self.r = r

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        self.cost = <double*> malloc(sizeof(double) * X.shape[2])
        self.cost_prev = <double*> malloc(sizeof(double) * X.shape[2])

        if self.cost == NULL or self.cost_prev == NULL:
            return -1

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

    def __reduce__(self):
        return self.__class__, (self.r, )

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return dtw_subsequence_distance(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.cost,
            self.cost_prev,
            NULL,
            return_index,
        )

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return dtw_subsequence_matches(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.cost,
            self.cost_prev,
            NULL,
            threshold,
            distances,
            indicies,
        )


cdef class AmercingDtwSubsequenceMetric(DtwSubsequenceMetric):
    cdef double p

    def __init__(self, double r=1.0, double p=1.0):
        super().__init__(r=r)
        self.p = p

    def __reduce__(self):
        return self.__class__, (self.r, self.p)

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return adtw_subsequence_distance(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.cost,
            self.cost_prev,
            self.p,
            return_index,
        )

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return adtw_subsequence_matches(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.cost,
            self.cost_prev,
            self.p,
            threshold,
            distances,
            indicies,
        )


cdef class WeightedDtwSubsequenceMetric(DtwSubsequenceMetric):

    cdef double g
    cdef double *weights

    def __init__(self, r=1.0, g=0.05):
        super().__init__(r=r)
        check_scalar(g, "g", float, min_val=0.0)
        self.g = g

    # Implicit super call
    def __cinit__(self, *args, **kwargs):
        self.weights = NULL

    cdef int reset(self, TSArray X) noexcept nogil:
        DtwSubsequenceMetric.reset(self, X)
        self.weights = <double*> malloc(sizeof(double) * X.shape[2])
        if self.weights == NULL:
            return -1

        cdef Py_ssize_t i
        for i in range(X.shape[2]):
            self.weights[i] = 1.0 / (1.0 + exp(-self.g * (i - X.shape[2] / 2.0)))

        return 0

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept nogil:
        DtwSubsequenceMetric._free(self)
        if self.weights != NULL:
            free(self.weights)
            self.weights = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.g)

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return dtw_subsequence_distance(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.cost,
            self.cost_prev,
            self.weights,
            return_index,
        )

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return dtw_subsequence_matches(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.cost,
            self.cost_prev,
            self.weights,
            threshold,
            distances,
            indicies,
        )


cdef class DerivativeDtwSubsequenceMetric(DtwSubsequenceMetric):
    cdef double *S_buffer
    cdef double *T_buffer

    def __cinit__(self, *args, **kwargs):
        self.T_buffer = NULL

    cdef int reset(self, TSArray X) noexcept nogil:
        DtwSubsequenceMetric.reset(self, X)
        self.T_buffer = <double*> malloc(sizeof(double) * X.shape[2])
        self.S_buffer = <double*> malloc(sizeof(double) * X.shape[2])
        if self.T_buffer == NULL or self.S_buffer == NULL:
            return -1

        return 0

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept nogil:
        DtwSubsequenceMetric._free(self)
        if self.T_buffer != NULL:
            free(self.T_buffer)
            self.T_buffer = NULL

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        average_slope(s, s_len, self.S_buffer)
        return ddtw_subsequence_distance(
            self.S_buffer,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.cost,
            self.cost_prev,
            NULL,
            self.T_buffer,
            return_index,
        )

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        average_slope(s, s_len, self.S_buffer)
        return ddtw_subsequence_matches(
            self.S_buffer,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.cost,
            self.cost_prev,
            NULL,
            self.T_buffer,
            threshold,
            distances,
            indicies,
        )

cdef class WeightedDerivativeDtwSubsequenceMetric(DerivativeDtwSubsequenceMetric):

    cdef double g
    cdef double *weights

    def __init__(self, r=1.0, g=0.05):
        super().__init__(r=r)
        check_scalar(g, "g", float, min_val=0.0)
        self.g = g

    # Implicit super call
    def __cinit__(self, *args, **kwargs):
        self.weights = NULL

    cdef int reset(self, TSArray X) noexcept nogil:
        DerivativeDtwSubsequenceMetric.reset(self, X)
        self.weights = <double*> malloc(sizeof(double) * X.shape[2])
        if self.weights == NULL:
            return -1

        cdef Py_ssize_t n_timestep = X.shape[2] - 2
        cdef Py_ssize_t i
        for i in range(n_timestep):
            self.weights[i] = 1.0 / (1.0 + exp(-self.g * (i - n_timestep / 2.0)))

        return 0

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept nogil:
        DerivativeDtwSubsequenceMetric._free(self)
        if self.weights != NULL:
            free(self.weights)
            self.weights = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.g)

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        average_slope(s, s_len, self.S_buffer)
        return ddtw_subsequence_distance(
            self.S_buffer,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.cost,
            self.cost_prev,
            self.weights,
            self.T_buffer,
            return_index,
        )

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        average_slope(s, s_len, self.S_buffer)
        return ddtw_subsequence_matches(
            self.S_buffer,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.cost,
            self.cost_prev,
            self.weights,
            self.T_buffer,
            threshold,
            distances,
            indicies,
        )


cdef class LcssSubsequenceMetric(SubsequenceMetric):

    cdef double *cost
    cdef double *cost_prev
    cdef double r
    cdef double epsilon

    def __init__(self, double r=1.0, double epsilon=1.0):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(epsilon, "epsilon", float, min_val=0, include_boundaries="neither")
        self.r = r
        self.epsilon = epsilon

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.epsilon)

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        self.cost = <double*> malloc(sizeof(double) * X.shape[2])
        self.cost_prev = <double*> malloc(sizeof(double) * X.shape[2])
        if self.cost == NULL or self.cost_prev == NULL:
            return -1

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return lcss_subsequence_distance(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.epsilon,
            self.cost,
            self.cost_prev,
            NULL,
            return_index,
        )

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return lcss_subsequence_matches(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.epsilon,
            self.cost,
            self.cost_prev,
            NULL,
            threshold,
            distances,
            indicies,
        )


cdef class EdrSubsequenceMetric(ScaledSubsequenceMetric):

    cdef double *cost
    cdef double *cost_prev
    cdef double r
    cdef double epsilon

    def __init__(self, double r=1.0, double epsilon=NAN):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)

        if not isnan(epsilon):
            check_scalar(
                epsilon, "epsilon", float, min_val=0, include_boundaries="neither"
            )

        self.r = r
        self.epsilon = epsilon

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.epsilon)

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        self.cost = <double*> malloc(sizeof(double) * X.shape[2])
        self.cost_prev = <double*> malloc(sizeof(double) * X.shape[2])
        if self.cost == NULL or self.cost_prev == NULL:
            return -1

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        if isnan(self.epsilon):
            epsilon = s_std / 4.0
        else:
            epsilon = self.epsilon

        return edr_subsequence_distance(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            epsilon,
            self.cost,
            self.cost_prev,
            NULL,
            return_index,
        )

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        if isnan(self.epsilon):
            epsilon = s_std / 4.0
        else:
            epsilon = self.epsilon

        return edr_subsequence_matches(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            epsilon,
            self.cost,
            self.cost_prev,
            NULL,
            threshold,
            distances,
            indicies,
        )


cdef class TweSubsequenceMetric(SubsequenceMetric):

    cdef double *cost
    cdef double *cost_prev
    cdef double r
    cdef double penalty
    cdef double stiffness

    def __init__(self, double r=1.0, double penalty=1.0, double stiffness=0.001):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(penalty, "penalty", float, min_val=0.0)
        check_scalar(
            stiffness, "stiffness", float, min_val=0.0, include_boundaries="neither"
        )
        self.r = r
        self.penalty = penalty
        self.stiffness = stiffness

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.penalty, self.stiffness)

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        self.cost = <double*> malloc(sizeof(double) * X.shape[2])
        self.cost_prev = <double*> malloc(sizeof(double) * X.shape[2])
        if self.cost == NULL or self.cost_prev == NULL:
            return -1

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return twe_subsequence_distance(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.penalty,
            self.stiffness,
            self.cost,
            self.cost_prev,
            NULL,
            return_index,
        )

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return twe_subsequence_matches(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.penalty,
            self.stiffness,
            self.cost,
            self.cost_prev,
            NULL,
            threshold,
            distances,
            indicies,
        )


cdef class MsmSubsequenceMetric(SubsequenceMetric):

    cdef double *cost
    cdef double *cost_prev
    cdef double *cost_y
    cdef double r
    cdef double c

    def __init__(self, double r=1.0, double c=1.0):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(c, "c", float, min_val=0)
        self.r = r
        self.c = c

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL
        self.cost_y = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.c)

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

        if self.cost_y != NULL:
            free(self.cost_y)
            self.cost_y = NULL

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        self.cost = <double*> malloc(sizeof(double) * X.shape[2])
        self.cost_prev = <double*> malloc(sizeof(double) * X.shape[2])
        self.cost_y = <double*> malloc(sizeof(double) * X.shape[2])
        if self.cost == NULL or self.cost_prev == NULL or self.cost_y == NULL:
            return -1

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return msm_subsequence_distance(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.c,
            self.cost,
            self.cost_prev,
            self.cost_y,
            return_index,
        )

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return msm_subsequence_matches(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.c,
            self.cost,
            self.cost_prev,
            self.cost_y,
            threshold,
            distances,
            indicies,
        )


cdef class ErpSubsequenceMetric(SubsequenceMetric):

    cdef double *cost
    cdef double *cost_prev
    cdef double *gX
    cdef double *gY
    cdef double r
    cdef double g

    def __init__(self, double r=1.0, double g=0.0):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(g, "g", float, min_val=0)
        self.r = r
        self.g = g

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL
        self.gX = NULL
        self.gY = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.g)

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept nogil:
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

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        self.cost = <double*> malloc(sizeof(double) * X.shape[2])
        self.cost_prev = <double*> malloc(sizeof(double) * X.shape[2])
        self.gX = <double*> malloc(sizeof(double) * X.shape[2])
        self.gY = <double*> malloc(sizeof(double) * X.shape[2])
        if (
            self.cost == NULL or
            self.cost_prev == NULL or
            self.gX == NULL or
            self.gY == NULL
        ):
            return -1

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return erp_subsequence_distance(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.g,
            self.gX,
            self.gY,
            self.cost,
            self.cost_prev,
            return_index,
        )

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return erp_subsequence_matches(
            s,
            s_len,
            x,
            x_len,
            _compute_r(s_len, self.r),
            self.g,
            self.gX,
            self.gY,
            self.cost,
            self.cost_prev,
            threshold,
            distances,
            indicies,
        )


cdef class DtwMetric(Metric):

    cdef double *cost
    cdef double *cost_prev
    cdef double *weights
    cdef double r

    def __init__(self, double r=1.0):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        self.r = r

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL
        self.weights = NULL

    def __reduce__(self):
        return self.__class__, (self.r, )

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) noexcept nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

        if self.weights != NULL:
            free(self.weights)
            self.weights = NULL

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(X.shape[2], Y.shape[2])
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        cdef double dist = dtw_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.cost,
            self.cost_prev,
            self.weights,
            INFINITY,
        )

        return sqrt(dist)

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist,
    ) noexcept nogil:
        cdef double dist = dtw_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.cost,
            self.cost_prev,
            self.weights,
            min_dist[0] * min_dist[0],
        )

        dist = sqrt(dist)
        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False

    @property
    def is_elastic(self):
        return True


cdef void average_slope(const double *q, Py_ssize_t len, double *d) noexcept nogil:
    cdef Py_ssize_t i, j
    j = 0
    for i in range(1, len - 1):
        d[j] = ((q[i] - q[i - 1]) + ((q[i + 1] - q[i - 1]) / 2)) / 2
        j += 1


cdef class DerivativeDtwMetric(DtwMetric):

    cdef double *d_x
    cdef double *d_y

    def __init__(self, double r=1.0):
        super().__init__(r=r)

    def __cinit__(self, *args, **kwargs):
        self.d_x = NULL
        self.d_y = NULL

    cdef void __free(self) noexcept nogil:
        DtwMetric.__free(self)
        if self.d_x != NULL:
            free(self.d_x)
            self.d_x = NULL

        if self.d_y != NULL:
            free(self.d_y)
            self.d_y = NULL

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil:
        DtwMetric.reset(self, Y, X)
        if min(X.shape[2], Y.shape[2]) < 3:
            return 0

        self.d_x = <double*> malloc(sizeof(double) * X.shape[2] - 2)
        self.d_y = <double*> malloc(sizeof(double) * Y.shape[2] - 2)

        if self.d_x == NULL or self.d_y == NULL:
            return -1

        return 0

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
    ) noexcept nogil:
        if min(x_len, y_len) < 3:
            return 0

        average_slope(x, x_len, self.d_x)
        average_slope(y, y_len, self.d_y)
        cdef double dist = dtw_distance(
            self.d_x,
            x_len - 2,
            self.d_y,
            y_len - 2,
            _compute_r(min(x_len, y_len), self.r),
            self.cost,
            self.cost_prev,
            self.weights,
            INFINITY,
        )

        return sqrt(dist)

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist
    ) noexcept nogil:
        if min(x_len, y_len) < 3:
            return 0

        average_slope(x, x_len, self.d_x)
        average_slope(y, y_len, self.d_y)

        cdef double dist = dtw_distance(
            self.d_x,
            x_len - 2,
            self.d_y,
            y_len - 2,
            _compute_r(min(x_len - 2, y_len - 2), self.r),
            self.cost,
            self.cost_prev,
            self.weights,
            min_dist[0] * min_dist[0],
        )

        dist = sqrt(dist)
        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False

cdef class WeightedDtwMetric(DtwMetric):

    cdef double g

    def __init__(self, double r=1.0, double g=0.05):
        super().__init__(r=r)
        check_scalar(g, "g", float, min_val=0.0)
        self.g = g

    def __reduce__(self):
        return self.__class__, (self.r, self.g)

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil:
        DtwMetric.reset(self, X, Y)
        cdef Py_ssize_t i
        cdef Py_ssize_t n_timestep = max(X.shape[2], Y.shape[2])

        self.weights = <double*> malloc(sizeof(double) * n_timestep)
        for i in range(n_timestep):
            self.weights[i] = 1.0 / (1.0 + exp(-self.g * (i - n_timestep / 2.0)))


cdef class AmercingDtwMetric(DtwMetric):
    cdef double p

    def __init__(self, double r=1.0, double p=1.0):
        super().__init__(r=r)
        check_scalar(p, "p", float)
        self.p = p

    def __reduce__(self):
        return self.__class__, (self.r, self.p)

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        cdef double dist = adtw_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.cost,
            self.cost_prev,
            self.p,
            INFINITY,
        )

        return sqrt(dist)

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist
    ) noexcept nogil:
        cdef double dist = adtw_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.cost,
            self.cost_prev,
            self.p,
            min_dist[0] * min_dist[0],
        )

        dist = sqrt(dist)
        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False


cdef class WeightedDerivativeDtwMetric(DerivativeDtwMetric):

    cdef double g

    def __init__(self, double r=1.0, double g=0.05):
        super().__init__(r=r)
        self.g = g

    def __reduce__(self):
        return self.__class__, (self.r, self.g)

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil:
        DerivativeDtwMetric.reset(self, X, Y)
        cdef Py_ssize_t i
        cdef Py_ssize_t n_timestep = max(X.shape[2] - 2, Y.shape[2] - 2)

        self.d_x = <double*> malloc(sizeof(double) * Y.shape[2] - 2)
        self.d_y = <double*> malloc(sizeof(double) * Y.shape[2] - 2)
        self.weights = <double*> malloc(sizeof(double) * n_timestep)

        if self.d_x == NULL or self.d_y == NULL or self.weights == NULL:
            return -1

        for i in range(0, n_timestep):
            self.weights[i] = 1.0 / (1.0 + exp(-self.g * (i - n_timestep / 2.0)))

        return 0


cdef class LcssMetric(Metric):

    cdef double *cost
    cdef double *cost_prev
    cdef double *weights
    cdef double r
    cdef double epsilon

    def __init__(self, double r=1.0, double epsilon=1.0, threshold=NAN):
        # TODO(1.4): remove deprecated
        if not isnan(threshold):
            warnings.warn(
                "The parameter threshold has been renamed to epsilon in 1.2 "
                "and will be removed in 1.4.",
                DeprecationWarning
            )
            epsilon = threshold

        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(epsilon, "epsilon", float, min_val=0, include_boundaries="neither")
        self.r = r
        self.epsilon = epsilon

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL
        self.weights = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.epsilon)

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) noexcept nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

        if self.weights != NULL:
            free(self.weights)
            self.weights = NULL

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(X.shape[2], Y.shape[2])
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        cdef double dist = lcss_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.epsilon,
            self.cost,
            self.cost_prev,
            self.weights,
            INFINITY,
        )

        return dist

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist
    ) noexcept nogil:
        cdef double dist = lcss_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.epsilon,
            self.cost,
            self.cost_prev,
            self.weights,
            min(x_len, y_len) - min_dist[0] * min(x_len, y_len)
            if isinf(min_dist[0]) == 0
            else INFINITY,
        )

        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False

    @property
    def is_elastic(self):
        return True


cdef class WeightedLcssMetric(LcssMetric):

    cdef double g

    def __init__(self, double r=1.0, epsilon=1.0, double g=0.05, double threshold=NAN):
        super().__init__(r=r, epsilon=epsilon, threshold=threshold)
        check_scalar(g, "g", float, min_val=0.0)
        self.g = g

    def __reduce__(self):
        return self.__class__, (self.r, self.epsilon, self.g)

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil:
        LcssMetric.reset(self, X, Y)
        cdef Py_ssize_t i
        cdef Py_ssize_t n_timestep = max(X.shape[2], Y.shape[2])

        self.weights = <double*> malloc(sizeof(double) * n_timestep)
        for i in range(n_timestep):
            self.weights[i] = 1.0 / (1.0 + exp(-self.g * (i - n_timestep / 2.0)))


cdef class ErpMetric(Metric):

    cdef double *cost
    cdef double *cost_prev
    cdef double *gX
    cdef double *gY

    cdef double r
    cdef double g

    def __init__(self, double r=1.0, double g=0.0):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(g, "g", float, min_val=0)
        self.r = r
        self.g = g

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.g)

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) noexcept nogil:
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

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(X.shape[2], Y.shape[2])
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)
        self.gX = <double*> malloc(sizeof(double) * X.shape[2])
        self.gY = <double*> malloc(sizeof(double) * Y.shape[2])

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        cdef double dist = erp_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.g,
            self.gX,
            self.gY,
            self.cost,
            self.cost_prev,
            INFINITY,
        )

        return dist

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist
    ) noexcept nogil:
        cdef double dist = erp_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.g,
            self.gX,
            self.gY,
            self.cost,
            self.cost_prev,
            min_dist[0],
        )

        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False

    @property
    def is_elastic(self):
        return True


cdef class EdrMetric(Metric):

    cdef double *cost
    cdef double *cost_prev
    cdef double *std_x
    cdef double *std_y

    cdef double r
    cdef double epsilon

    def __init__(self, double r=1.0, double epsilon=NAN, double threshold=NAN):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)

        # TODO(1.4): remove deprecated
        if not isnan(threshold):
            warnings.warn(
                "The parameter threshold has been renamed to epsilon in 1.2 "
                "and will be removed in 1.4.",
                DeprecationWarning
            )
            epsilon = threshold

        if not isnan(epsilon):
            check_scalar(
                epsilon, "epsilon", float, min_val=0, include_boundaries="neither"
            )

        self.r = r
        self.epsilon = epsilon

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL
        self.std_x = NULL
        self.std_y = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.epsilon)

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) noexcept nogil:
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

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(X.shape[2], Y.shape[2])
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)

        # Store standard deviation for all samples and all dimensions:
        # [x[0,0], x[0,1] ... x[0, m], x[1,0], x[1,1], ... x[d, m]]
        self.std_x = <double*> malloc(sizeof(double) * X.shape[0] * X.shape[1])
        self.std_y = <double*> malloc(sizeof(double) * Y.shape[0] * Y.shape[1])

        cdef double mean, std
        cdef Py_ssize_t i, d
        if isnan(self.epsilon):
            for i in range(X.shape[0]):
                for d in range(X.shape[1]):
                    fast_mean_std(&X[i, d, 0], X.shape[2], &mean, &std)
                    self.std_x[d * X.shape[0] + i] = std

            for i in range(Y.shape[0]):
                for d in range(Y.shape[1]):
                    fast_mean_std(&Y[i, d, 0], Y.shape[2], &mean, &std)
                    self.std_y[d * Y.shape[0] + i] = std

    cdef double distance(
        self,
        TSArray X,
        Py_ssize_t x_index,
        TSArray Y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) noexcept nogil:
        if isnan(self.epsilon):
            epsilon = max(
                self.std_x[dim * X.shape[0] + x_index],
                self.std_y[dim * Y.shape[0] + y_index],
            ) / 4.0
        else:
            epsilon = self.epsilon

        return edr_distance(
            &X[x_index, dim, 0],
            X.shape[2],
            &Y[y_index, dim, 0],
            Y.shape[2],
            _compute_r(min(X.shape[2], Y.shape[2]), self.r),
            epsilon,
            self.cost,
            self.cost_prev,
            NULL,
            INFINITY,
        )

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
    ) noexcept nogil:
        cdef double mean, std_x, std_y, epsilon
        if isnan(self.epsilon):
            fast_mean_std(x, x_len, &mean, &std_x)
            fast_mean_std(y, y_len, &mean, &std_y)
            epsilon = max(std_x, std_y) / 4.0
        else:
            epsilon = self.epsilon

        return edr_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            epsilon,
            self.cost,
            self.cost_prev,
            NULL,
            INFINITY,
        )

    cdef bint eadistance(
        self,
        TSArray X,
        Py_ssize_t x_index,
        TSArray Y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
        double *min_dist,
    ) noexcept nogil:
        if isnan(self.epsilon):
            epsilon = max(
                self.std_x[dim * X.shape[0] + x_index],
                self.std_y[dim * Y.shape[0] + y_index],
            ) / 4.0
        else:
            epsilon = self.epsilon

        cdef double dist = edr_distance(
            &X[x_index, dim, 0],
            X.shape[2],
            &Y[y_index, dim, 0],
            Y.shape[2],
            _compute_r(min(X.shape[2], X.shape[2]), self.r),
            epsilon,
            self.cost,
            self.cost_prev,
            NULL,
            min_dist[0] * max(X.shape[2], Y.shape[2]),
        )

        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist
    ) noexcept nogil:
        cdef double mean, std_x, std_y, epsilon
        if isnan(self.epsilon):
            fast_mean_std(x, x_len, &mean, &std_x)
            fast_mean_std(y, y_len, &mean, &std_y)
            epsilon = max(std_x, std_y) / 4.0
        else:
            epsilon = self.epsilon

        cdef double dist = edr_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            epsilon,
            self.cost,
            self.cost_prev,
            NULL,
            min_dist[0] * max(x_len, y_len),
        )

        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False

    @property
    def is_elastic(self):
        return True


cdef class MsmMetric(Metric):

    cdef double *cost
    cdef double *cost_prev
    cdef double *cost_y

    cdef double r
    cdef double c

    def __init__(self, double r=1.0, double c=1.0):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(c, "c", float, min_val=0)
        self.r = r
        self.c = c

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL
        self.cost_y = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.c)

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) noexcept nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

        if self.cost_y != NULL:
            free(self.cost_y)
            self.cost_y = NULL

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(X.shape[2], Y.shape[2])
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_y = <double*> malloc(sizeof(double) * X.shape[2])

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return msm_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.c,
            self.cost,
            self.cost_prev,
            self.cost_y,
            INFINITY,
        )

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist
    ) noexcept nogil:
        cdef double dist = msm_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.c,
            self.cost,
            self.cost_prev,
            self.cost_y,
            min_dist[0],
        )

        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False

    @property
    def is_elastic(self):
        return True


cdef class TweMetric(Metric):

    cdef double *cost
    cdef double *cost_prev
    cdef double r
    cdef double penalty
    cdef double stiffness

    def __init__(
        self,
        double r=1.0,
        double penalty=1.0,
        double stiffness=0.001,
    ):
        check_scalar(r, "r", float, min_val=0.0, max_val=1.0)
        check_scalar(penalty, "penalty", float, min_val=0.0)
        check_scalar(
            stiffness, "stiffness", float, min_val=0.0, include_boundaries="neither"
        )
        self.r = r
        self.penalty = penalty
        self.stiffness = stiffness

    def __cinit__(self, *args, **kwargs):
        self.cost = NULL
        self.cost_prev = NULL

    def __reduce__(self):
        return self.__class__, (self.r, self.penalty, self.stiffness)

    def __dealloc__(self):
        self.__free()

    cdef void __free(self) noexcept nogil:
        if self.cost != NULL:
            free(self.cost)
            self.cost = NULL

        if self.cost_prev != NULL:
            free(self.cost_prev)
            self.cost_prev = NULL

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil:
        self.__free()
        cdef Py_ssize_t n_timestep = max(X.shape[2], Y.shape[2])
        self.cost = <double*> malloc(sizeof(double) * n_timestep)
        self.cost_prev = <double*> malloc(sizeof(double) * n_timestep)

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return twe_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.penalty,
            self.stiffness,
            self.cost,
            self.cost_prev,
            INFINITY,
        )

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist
    ) noexcept nogil:
        cdef double dist = twe_distance(
            x,
            x_len,
            y,
            y_len,
            _compute_r(min(x_len, y_len), self.r),
            self.penalty,
            self.stiffness,
            self.cost,
            self.cost_prev,
            min_dist[0],
        )

        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False

    @property
    def is_elastic(self):
        return True
