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

cimport numpy as np

from .._data cimport TSDatabase


cdef struct TSView:
    Py_ssize_t index  # the index of the shapelet sample
    Py_ssize_t start  # the start position
    Py_ssize_t length  # the length of the shapelet
    Py_ssize_t dim  # the dimension of the shapelet
    double mean  # the mean of the shapelet
    double std  # the stanard devision
    void *extra


cdef struct TSCopy:
    Py_ssize_t length
    Py_ssize_t dim
    double mean
    double std
    int ts_index
    int ts_start
    double *data
    void *extra


cdef class DistanceMeasure:
    cdef Py_ssize_t n_timestep

    cdef int init(self, TSDatabase *td) nogil

    cdef int init_ts_view(
        self,
        TSDatabase *td,
        TSView *ts_view,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil

    cdef int init_ts_copy_from_obj(
        self,
        TSCopy *ts_copy,
        object obj,
    )

    cdef object object_from_ts_copy(
        self, 
        TSCopy *ts_copy
    )

    cdef int init_ts_copy(
        self,
        TSCopy *ts_copy,
        TSView *s,
        TSDatabase *td,
    ) nogil

    cdef int ts_copy_sub_matches(
        self,
        TSCopy *ts_copy,
        TSDatabase *td,
        Py_ssize_t t_index,
        double threshold,
        Py_ssize_t** matches,
        double** distances,
        Py_ssize_t *n_matches,
    ) nogil except -1

    cdef double ts_copy_sub_distance(
        self,
        TSCopy *ts_copy,
        TSDatabase *td,
        Py_ssize_t t_index,
        Py_ssize_t *return_index= *,
    ) nogil

    cdef double ts_view_sub_distance(
        self,
        TSView *ts_view,
        TSDatabase *td,
        Py_ssize_t t_index,
    ) nogil

    cdef void ts_view_sub_distances(
        self,
        TSView *ts_view,
        TSDatabase *td,
        Py_ssize_t *samples,
        double *distances,
        Py_ssize_t n_samples,
    ) nogil

    cdef double ts_copy_distance(
        self,
        TSCopy *ts_copy,
        TSDatabase *td,
        Py_ssize_t t_index,
    ) nogil

    cdef bint support_unaligned(self) nogil


cdef class ScaledDistanceMeasure(DistanceMeasure):
    pass


cdef void ts_view_init(TSView *s) nogil


cdef void ts_view_free(TSView *shapelet_info) nogil


cdef int ts_copy_init(
    TSCopy *ts_copy,
    Py_ssize_t dim,
    Py_ssize_t length,
    double mean,
    double std,
) nogil


cdef void ts_copy_free(TSCopy *shapelet) nogil


cpdef DistanceMeasure get_distance_measure(
    Py_ssize_t n_timestep,
    object metric,
    dict metric_params=*,
)
