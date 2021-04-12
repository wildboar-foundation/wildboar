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
#
# Authors: Isak Samsten

import numpy as np

cimport numpy as np
from libc.math cimport INFINITY, sqrt
from libc.stdlib cimport free, malloc

from .._data cimport TSDatabase
from .._utils cimport realloc_array
from ._distance cimport DistanceMeasure, ScaledDistanceMeasure, TSCopy, TSView


cdef class ScaledEuclideanDistance(ScaledDistanceMeasure):
    cdef double *X_buffer


    def __cinit__(self, Py_ssize_t n_timestep, *args, **kwargs):
        super().__init__(n_timestep)
        self.X_buffer = <double*> malloc(sizeof(double) * n_timestep * 2)


    cdef double ts_copy_sub_distance(
        self,
        TSCopy *ts_copy,
        TSDatabase *td,
        Py_ssize_t t_index,
        Py_ssize_t *return_index = NULL,
    ) nogil:
        cdef Py_ssize_t sample_offset = (
            t_index * td.sample_stride + ts_copy.dim * td.dim_stride
        )

        return scaled_euclidean_distance(
            0,
            1,
            ts_copy.length,
            ts_copy.mean,
            ts_copy.std,
            ts_copy.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            self.X_buffer,
            return_index,
        )


    def __dealloc__(self):
        free(self.X_buffer)


    cdef double ts_view_sub_distance(
        self,
        TSView *ts_view,
        TSDatabase *td,
        Py_ssize_t t_index,
    ) nogil:
        cdef Py_ssize_t sample_offset = (
            t_index * td.sample_stride + ts_view.dim * td.dim_stride
        )
        cdef Py_ssize_t shapelet_offset = (
            ts_view.index * td.sample_stride +
            ts_view.dim * td.dim_stride +
            ts_view.start * td.timestep_stride
        )
        return scaled_euclidean_distance(
            shapelet_offset,
            td.timestep_stride,
            ts_view.length,
            ts_view.mean,
            ts_view.std,
            td.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            self.X_buffer,
            NULL,
        )


    cdef int ts_copy_matches(
        self,
        TSCopy *ts_copy,
        TSDatabase *td,
        Py_ssize_t t_index,
        double threshold,
        Py_ssize_t** matches,
        double** distances,
        Py_ssize_t *n_matches,
    ) nogil except -1:
        cdef Py_ssize_t sample_offset = (
            t_index * td.sample_stride + ts_copy.dim * td.dim_stride
        )

        return scaled_euclidean_distance_matches(
            0,
            1,
            ts_copy.length,
            ts_copy.mean,
            ts_copy.std,
            ts_copy.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            self.X_buffer,
            threshold,
            distances,
            matches,
            n_matches,
        )


cdef class EuclideanDistance(DistanceMeasure):

    cdef double ts_copy_sub_distance(
        self,
        TSCopy *ts_copy,
        TSDatabase *td,
        Py_ssize_t t_index,
        Py_ssize_t *return_index = NULL,
    ) nogil:
        cdef Py_ssize_t sample_offset = (
            t_index * td.sample_stride + ts_copy.dim * td.dim_stride
        )

        return euclidean_distance(
            0,
            1,
            ts_copy.length,
            ts_copy.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            return_index,
        )


    cdef double ts_view_sub_distance(
        self,
        TSView *ts_view,
        TSDatabase *td,
        Py_ssize_t t_index,
    ) nogil:
        cdef Py_ssize_t sample_offset = (
            t_index * td.sample_stride + ts_view.dim * td.dim_stride
        )
        cdef Py_ssize_t shapelet_offset = (
            ts_view.index * td.sample_stride +
            ts_view.dim * td.dim_stride +
            ts_view.start * td.timestep_stride
        )
        return euclidean_distance(
            shapelet_offset,
            td.timestep_stride,
            ts_view.length,
            td.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            NULL,
        )


    cdef int ts_copy_matches(
        self,
        TSCopy *ts_copy,
        TSDatabase *td,
        Py_ssize_t t_index,
        double threshold,
        Py_ssize_t** matches,
        double** distances,
        Py_ssize_t *n_matches,
    ) nogil except -1:
        cdef Py_ssize_t sample_offset = (
            t_index * td.sample_stride + ts_copy.dim * td.dim_stride
        )
        return euclidean_distance_matches(
            0,
            1,
            ts_copy.length,
            ts_copy.data,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            threshold,
            distances,
            matches,
            n_matches,
        )


cdef double scaled_euclidean_distance(
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
    double *X_buffer,
    Py_ssize_t *index,
) nogil:
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
        current_value = T[t_offset + t_stride * i]
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
            dist = inner_scaled_euclidean_distance(s_offset, s_length, s_mean, s_std,
                                                   j, mean, std, S, s_stride,
                                                   X_buffer, min_dist)

            if dist < min_dist:
                min_dist = dist
                if index != NULL:
                    index[0] = (i + 1) - s_length

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return sqrt(min_dist)


cdef double inner_scaled_euclidean_distance(
    Py_ssize_t offset,
    Py_ssize_t length,
    double s_mean,
    double s_std,
    Py_ssize_t j,
    double mean,
    double std,
    double *X,
    Py_ssize_t timestep_stride,
    double *X_buffer,
    double min_dist,
) nogil:
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
        x = (X[offset + timestep_stride * i] - s_mean) / s_std
        x -= (X_buffer[i + j] - mean) / std
        dist += x * x

    return dist


cdef double euclidean_distance(
    Py_ssize_t s_offset,
    Py_ssize_t s_stride,
    Py_ssize_t s_length,
    double *S,
    Py_ssize_t t_offset,
    Py_ssize_t t_stride,
    Py_ssize_t t_length,
    double *T,
    Py_ssize_t *index,
) nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist >= min_dist:
                break

            x = T[t_offset + t_stride * i + j]
            x -= S[s_offset + s_stride * j]
            dist += x * x

        if dist < min_dist:
            min_dist = dist
            if index != NULL:
                index[0] = i

    return sqrt(min_dist)


cdef int euclidean_distance_matches(
    Py_ssize_t s_offset,
    Py_ssize_t s_stride,
    Py_ssize_t s_length,
    double *S,
    Py_ssize_t t_offset,
    Py_ssize_t t_stride,
    Py_ssize_t t_length,
    double *T,
    double threshold,
    double **distances,
    Py_ssize_t **matches,
    Py_ssize_t *n_matches,
) nogil except -1:
    cdef double dist = 0
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double x

    matches[0] = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * capacity)
    distances[0] = <double*> malloc(sizeof(double) * capacity)
    n_matches[0] = 0

    threshold = threshold * threshold
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist > threshold:
                break

            x = T[t_offset + t_stride * i + j]
            x -= S[s_offset + s_stride * j]
            dist += x * x
        if dist <= threshold:
            tmp_capacity = capacity
            realloc_array(<void**> matches, n_matches[0], sizeof(Py_ssize_t), &tmp_capacity)
            realloc_array(<void**> distances, n_matches[0], sizeof(double), &capacity)
            matches[0][n_matches[0]] = i
            distances[0][n_matches[0]] = sqrt(dist)
            n_matches[0] += 1

    return 0


cdef int scaled_euclidean_distance_matches(
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
   double *X_buffer,
   double threshold,
   double** distances,
   Py_ssize_t** matches,
   Py_ssize_t *n_matches,
) nogil except -1:
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0

    cdef double ex = 0
    cdef double ex2 = 0

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t buffer_pos
    cdef Py_ssize_t capacity = 1
    cdef Py_ssize_t tmp_capacity

    matches[0] = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * capacity)
    distances[0] = <double*> malloc(sizeof(double) * capacity)
    n_matches[0] = 0

    threshold = threshold * threshold

    for i in range(t_length):
        current_value = T[t_offset + t_stride * i]
        ex += current_value
        ex2 += current_value * current_value

        buffer_pos = i % s_length
        X_buffer[buffer_pos] = current_value
        X_buffer[buffer_pos + s_length] = current_value
        if i >= s_length - 1:
            j = (i + 1) % s_length
            mean = ex / s_length
            ex2 = ex2 / s_length - mean * mean
            if ex2 > 0:
                std = sqrt(ex2)
            else:
                std = 1.0
            dist = inner_scaled_euclidean_distance(
                s_offset, s_length, s_mean, s_std, j, mean, std, S, s_stride,
                X_buffer, threshold)

            if dist <= threshold:
                tmp_capacity = capacity
                realloc_array(
                    <void**> matches, n_matches[0], sizeof(Py_ssize_t), &tmp_capacity)
                realloc_array(
                    <void**> distances, n_matches[0], sizeof(double), &capacity)

                matches[0][n_matches[0]] = (i + 1) - s_length
                distances[0][n_matches[0]] = sqrt(dist)

                n_matches[0] += 1

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return 0
