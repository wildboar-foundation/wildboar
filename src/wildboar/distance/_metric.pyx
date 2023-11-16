# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
from libc.math cimport INFINITY, M_PI, acos, cos, fabs, pow, sqrt, isinf
from libc.stdlib cimport free, malloc

from ..utils cimport TSArray
from ..utils._misc cimport realloc_array
from ..utils._stats cimport IncStats, inc_stats_add, inc_stats_init, inc_stats_variance
from ._cdistance cimport (
    EPSILON,
    Metric,
    ScaledSubsequenceMetric,
    Subsequence,
    SubsequenceMetric,
    SubsequenceView,
)


cdef class EuclideanSubsequenceMetric(SubsequenceMetric):
    def __init__(self):
        pass

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
        return euclidean_distance(s, s_len, x, x_len, INFINITY, return_index)

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
        return euclidean_distance_matches(
            s, s_len, x, x_len, threshold, distances, indicies,
        )


cdef class NormalizedEuclideanSubsequenceMetric(SubsequenceMetric):
    def __init__(self):
        pass

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
        return normalized_euclidean_distance(s, s_len, x, x_len, return_index)

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
        return normalized_euclidean_distance_matches(
            s, s_len, x, x_len, threshold, distances, indicies,
        )


cdef class ScaledEuclideanSubsequenceMetric(ScaledSubsequenceMetric):
    cdef double *X_buffer

    def __init__(self):
        pass

    def __cinit__(self):
        self.X_buffer = NULL

    def __dealloc__(self):
        if self.X_buffer != NULL:
            free(self.X_buffer)
            self.X_buffer = NULL

    def __reduce__(self):
        return self.__class__, ()

    cdef int reset(self, TSArray X) noexcept nogil:
        if self.X_buffer != NULL:
            free(self.X_buffer)

        self.X_buffer = <double*> malloc(sizeof(double) * X.shape[2] * 2)
        if self.X_buffer == NULL:
            return -1

        return 0

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
        return scaled_euclidean_distance(
            s,
            s_len,
            s_mean,
            s_std,
            x,
            x_len,
            self.X_buffer,
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
        return scaled_euclidean_distance_matches(
            s,
            s_len,
            s_mean,
            s_std,
            x,
            x_len,
            self.X_buffer,
            threshold,
            distances,
            indicies
        )


cdef class ManhattanSubsequenceMetric(SubsequenceMetric):
    def __init__(self):
        pass

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
        return manhattan_distance(s, s_len, x, x_len, INFINITY, return_index)

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
        return manhattan_distance_matches(
            s, s_len, x, x_len, threshold, distances, indicies
        )


cdef class MinkowskiSubsequenceMetric(SubsequenceMetric):
    cdef double p

    def __init__(self, double p=2):
        self.p = p

    def __reduce__(self):
        return self.__class__, (self.p, )

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
        return minkowski_distance(self.p, s, s_len, x, x_len, INFINITY, return_index)

    cdef Py_ssize_t _matches(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return minkowski_distance_matches(
            self.p, s, s_len, x, x_len, threshold, distances, indicies
        )


cdef class ChebyshevSubsequenceMetric(SubsequenceMetric):
    def __init__(self):
        pass

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
        return chebyshev_distance(s, s_len, x, x_len, INFINITY, return_index)

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
        return chebyshev_distance_matches(
            s, s_len, x, x_len, threshold, distances, indicies
        )


cdef class CosineSubsequenceMetric(SubsequenceMetric):
    def __init__(self):
        pass

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
        return 1 - cosine_similarity(s, s_len, x, x_len, return_index)

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
        cdef Py_ssize_t i
        cdef Py_ssize_t n_matches = cosine_similarity_matches(
            s, s_len, x, x_len, 1 - threshold, distances, indicies
        )

        for i in range(n_matches):
            distances[i] = 1 - distances[i]

        return n_matches


cdef class AngularSubsequenceMetric(SubsequenceMetric):
    def __init__(self):
        pass

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
        cdef double cosine = cosine_similarity(s, s_len, x, x_len, return_index)

        # Edge-case where, due to floating point issues, the cosine is larger 1
        # or smaller than -1 we return the min and max value respectivley.
        if cosine > 1:
            return 0
        elif cosine < -1:
            return 1
        else:
            return acos(cosine) / M_PI

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
        cdef Py_ssize_t i
        threshold = cos(threshold * M_PI) if not isinf(threshold) else -INFINITY
        cdef Py_ssize_t n_matches = cosine_similarity_matches(
            s, s_len, x, x_len, threshold, distances, indicies
        )
        cdef double cosine

        for i in range(n_matches):
            cosine = distances[i]
            if cosine > 1:
                distances[i] = 0
            elif cosine < -1:
                distances[i] = 1
            else:
                distances[i] = acos(cosine) / M_PI

        return n_matches


cdef class EuclideanMetric(Metric):
    def __init__(self):
        pass

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return euclidean_distance(x, x_len, y, y_len, INFINITY, NULL)

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist,
    ) noexcept nogil:
        cdef double dist = euclidean_distance(
            x, x_len, y, y_len, min_dist[0] * min_dist[0], NULL
        )
        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False


cdef class NormalizedEuclideanMetric(Metric):
    def __init__(self):
        pass

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return normalized_euclidean_distance(x, x_len, y, y_len, NULL)


cdef class ManhattanMetric(Metric):
    def __init__(self):
        pass

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return manhattan_distance(x, x_len, y, y_len, INFINITY, NULL)

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist,
    ) noexcept nogil:
        cdef double dist = manhattan_distance(
            x, x_len, y, y_len, min_dist[0], NULL
        )
        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False

cdef class MinkowskiMetric(Metric):
    cdef double p

    def __init__(self, double p=2):
        self.p = p

    def __reduce__(self):
        return self.__class__, (self.p, )

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return minkowski_distance(self.p, x, x_len, y, y_len, INFINITY, NULL)

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist,
    ) noexcept nogil:
        cdef double dist = minkowski_distance(
            self.p, x, x_len, y, y_len, pow(min_dist[0], self.p), NULL
        )

        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False


cdef class ChebyshevMetric(Metric):

    def __init__(self):
        pass

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return chebyshev_distance(x, x_len, y, y_len, INFINITY, NULL)

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist,
    ) noexcept nogil:
        cdef double dist = chebyshev_distance(
            x, x_len, y, y_len, min_dist[0], NULL
        )
        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False


cdef class CosineMetric(Metric):

    def __init__(self):
        pass

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return 1 - cosine_similarity(x, x_len, y, y_len, NULL)

# https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
cdef class AngularMetric(Metric):

    def __init__(self):
        pass

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        cdef double cosine = cosine_similarity(x, x_len, y, y_len, NULL)
        if cosine > 1:
            return 0
        elif cosine < -1:
            return 1
        else:
            return acos(cosine) / M_PI


cdef double scaled_euclidean_distance(
    const double *S,
    Py_ssize_t s_length,
    double s_mean,
    double s_std,
    const double *T,
    Py_ssize_t t_length,
    double *X_buffer,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef double ex = 0
    cdef double ex2 = 0
    cdef double tmp

    cdef Py_ssize_t i
    cdef Py_ssize_t j
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
            mean = ex / s_length
            tmp = ex2 / s_length - mean * mean
            if tmp > 0:
                std = sqrt(tmp)
            else:
                std = 1.0
            dist = inner_scaled_euclidean_distance(
                s_length,
                s_mean,
                s_std,
                j,
                mean,
                std,
                S,
                X_buffer,
                min_dist,
            )

            if dist < min_dist:
                min_dist = dist
                if index != NULL:
                    index[0] = (i + 1) - s_length

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return sqrt(min_dist)


cdef double inner_scaled_euclidean_distance(
    Py_ssize_t length,
    double s_mean,
    double s_std,
    Py_ssize_t j,
    double mean,
    double std,
    const double *X,
    double *X_buffer,
    double min_dist,
) noexcept nogil:
    # Compute the distance between the shapelet (starting at `offset`
    # and ending at `offset + length` normalized with `s_mean` and
    # `s_std` with the shapelet in `X_buffer` starting at `0` and
    # ending at `length` normalized with `mean` and `std`
    cdef double dist = 0
    cdef double x
    cdef Py_ssize_t i

    for i in range(length):
        if dist >= min_dist:
            break
        x = (X[i] - s_mean) / s_std
        x -= (X_buffer[i + j] - mean) / std
        dist += x * x

    return dist


cdef double euclidean_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double min_dist,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double dist = 0

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist >= min_dist:
                break

            x = T[i + j]
            x -= S[j]
            dist += x * x

        if dist < min_dist:
            min_dist = dist
            if index != NULL:
                index[0] = i

    return sqrt(min_dist)

# PTDS (https://stats.stackexchange.com/users/68112/ptds),
#   Definition of normalized Euclidean distance, URL (version: 2021-09-27):
#   https://stats.stackexchange.com/q/498753
cdef double normalized_euclidean_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t *index,
) noexcept nogil:
    cdef IncStats S_stats, T_stats, ST_stats
    cdef double s, t, dist
    cdef double min_dist = INFINITY
    cdef Py_ssize_t i, j
    for i in range(t_length - s_length + 1):
        inc_stats_init(&S_stats)
        inc_stats_init(&T_stats)
        inc_stats_init(&ST_stats)
        for j in range(s_length):
            t = T[i + j]
            s = S[j]
            inc_stats_add(&S_stats, 1.0, s)
            inc_stats_add(&T_stats, 1.0, t)
            inc_stats_add(&ST_stats, 1.0, s - t)

        dist = inc_stats_variance(&S_stats) + inc_stats_variance(&T_stats)
        if dist > 0:
            dist = inc_stats_variance(&ST_stats) / dist

        if dist < min_dist:
            min_dist = dist
            if index != NULL:
                index[0] = i

    return sqrt(0.5 * min_dist)


cdef Py_ssize_t normalized_euclidean_distance_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double threshold,
    double *distances,
    Py_ssize_t *matches,
) except -1 nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i, j
    cdef double s, t
    cdef IncStats S_stats, T_stats, ST_stats
    cdef Py_ssize_t n_matches = 0

    for i in range(t_length - s_length + 1):
        inc_stats_init(&S_stats)
        inc_stats_init(&T_stats)
        inc_stats_init(&ST_stats)
        for j in range(s_length):
            t = T[i + j]
            s = S[j]
            inc_stats_add(&S_stats, 1.0, s)
            inc_stats_add(&T_stats, 1.0, t)
            inc_stats_add(&ST_stats, 1.0, s - t)

        dist = inc_stats_variance(&S_stats) + inc_stats_variance(&T_stats)
        if dist > 0:
            dist = sqrt(0.5 * inc_stats_variance(&ST_stats) / dist)

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = dist
            n_matches += 1

    return n_matches


cdef Py_ssize_t euclidean_distance_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double threshold,
    double *distances,
    Py_ssize_t *matches,
) except -1 nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    cdef Py_ssize_t n_matches = 0

    threshold = threshold * threshold
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist > threshold:
                break

            x = T[i + j]
            x -= S[j]
            dist += x * x
        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = sqrt(dist)
            n_matches += 1

    return n_matches


cdef Py_ssize_t scaled_euclidean_distance_matches(
   const double *S,
   Py_ssize_t s_length,
   double s_mean,
   double s_std,
   const double *T,
   Py_ssize_t t_length,
   double *X_buffer,
   double threshold,
   double *distances,
   Py_ssize_t *matches,
) noexcept nogil:
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0

    cdef double ex = 0
    cdef double ex2 = 0
    cdef double tmp

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t buffer_pos
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t n_matches = 0

    n_matches = 0
    threshold = threshold * threshold
    for i in range(t_length):
        current_value = T[i]
        ex += current_value
        ex2 += current_value * current_value

        buffer_pos = i % s_length
        X_buffer[buffer_pos] = current_value
        X_buffer[buffer_pos + s_length] = current_value
        if i >= s_length - 1:
            j = (i + 1) % s_length
            mean = ex / s_length
            tmp = ex2 / s_length - mean * mean
            if tmp > 0:
                std = sqrt(tmp)
            else:
                std = 1.0
            dist = inner_scaled_euclidean_distance(
                s_length,
                s_mean,
                s_std,
                j,
                mean,
                std,
                S,
                X_buffer,
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


cdef double manhattan_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double min_dist,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double dist = 0

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist >= min_dist:
                break

            dist += fabs(T[i + j] - S[j])

        if dist < min_dist:
            min_dist = dist
            if index != NULL:
                index[0] = i

    return min_dist


cdef Py_ssize_t manhattan_distance_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double threshold,
    double *distances,
    Py_ssize_t *matches,
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    cdef Py_ssize_t n_matches = 0

    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist > threshold:
                break

            dist += fabs(T[i + j] - S[j])

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = dist
            n_matches += 1

    return n_matches

cdef double minkowski_distance(
    double p,
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double min_dist,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist >= min_dist:
                break

            dist += pow(fabs(T[i + j] - S[j]), p)

        if dist < min_dist:
            min_dist = dist
            if index != NULL:
                index[0] = i

    return pow(min_dist, 1.0 / p)


cdef Py_ssize_t minkowski_distance_matches(
    double p,
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double threshold,
    double *distances,
    Py_ssize_t *matches,
) noexcept nogil:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    cdef Py_ssize_t n_matches = 0

    threshold = pow(threshold, p)
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist > threshold:
                break

            dist += pow(fabs(T[i + j] - S[j]), p)

        if dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = pow(dist, 1.0 / p)
            n_matches += 1

    return n_matches


cdef double chebyshev_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double min_dist,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double c = 0
    cdef double max_dist

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    for i in range(t_length - s_length + 1):
        max_dist = -INFINITY
        for j in range(s_length):
            if max_dist >= min_dist:
                break

            c = fabs(T[i + j] - S[j])
            max_dist = c if c > max_dist else max_dist

        if max_dist < min_dist:
            min_dist = max_dist
            if index != NULL:
                index[0] = i

    return min_dist


cdef Py_ssize_t chebyshev_distance_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double threshold,
    double *distances,
    Py_ssize_t *matches,
) noexcept nogil:
    cdef double c = 0
    cdef double max_dist
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    cdef Py_ssize_t n_matches = 0

    for i in range(t_length - s_length + 1):
        max_dist = -INFINITY
        for j in range(s_length):
            if max_dist > threshold:
                break

            c = fabs(T[i + j] - S[j])
            max_dist = c if c > max_dist else max_dist

        if max_dist <= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = max_dist
            n_matches += 1

    return n_matches


cdef double cosine_similarity(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t *index,
) noexcept nogil:
    cdef double prod = 0, sim
    cdef double s_norm, t_norm
    cdef double max_sim = -INFINITY

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    for i in range(t_length - s_length + 1):
        prod = 0
        s_norm = 0
        t_norm = 0
        for j in range(s_length):
            prod += T[i + j] * S[j]
            s_norm += pow(S[j], 2)
            t_norm += pow(T[i + j], 2)

        sim = (sqrt(s_norm) * sqrt(t_norm))
        if sim <= EPSILON:
            sim = 0
        else:
            sim = prod / sim

        if sim > max_sim:
            max_sim = sim
            if index != NULL:
                index[0] = i

    return max_sim


cdef Py_ssize_t cosine_similarity_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double threshold,
    double *distances,
    Py_ssize_t *matches,
) noexcept nogil:
    cdef double prod = 0, sim, s_norm, t_norm
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    cdef Py_ssize_t n_matches = 0

    for i in range(t_length - s_length + 1):
        prod = 0
        s_norm = 0
        t_norm = 0
        for j in range(s_length):
            prod += T[i + j] * S[j]
            s_norm += pow(S[j], 2)
            t_norm += pow(T[i + j], 2)

        sim = prod / (sqrt(s_norm) * sqrt(t_norm))
        if sim >= threshold:
            if matches != NULL:
                matches[n_matches] = i
            distances[n_matches] = sim
            n_matches += 1

    return n_matches
