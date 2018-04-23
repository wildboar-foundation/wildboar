# This file is part of pypf
#
# pypf is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pypf is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

# Authors: Isak Karlsson

from pypf._distance cimport DistanceMeasure
from pypf._distance cimport ScaledDistanceMeasure

cdef struct CircularArray:
    size_t* queue
    size_t size
    size_t capacity
    size_t front
    size_t back

cdef void init_circular(CircularArray* c, size_t capacity) nogil

cdef void destroy_cicular(CircularArray* c) nogil

cdef void cirular_push_back(CircularArray* c, size_t v) nogil

cdef void circular_pop_front(CircularArray* c) nogil

cdef void cicular_pop_back(CircularArray* c) nogil

cdef size_t circular_front(CircularArray* c) nogil

cdef size_t circular_back(CircularArray* c) nogil

cdef size_t circular_size(CircularArray* c) nogil


# cdef double scaled_dtw_distance(size_t s_offset,
#                                 size_t s_stride,
#                                 size_t s_length,
#                                 double s_mean,
#                                 double s_std,
#                                 double* S,
#                                 size_t t_offset,
#                                 size_t t_stride,
#                                 size_t t_length,
#                                 double* T,
#                                 double* X_buffer,
#                                 size_t* index) nogil


# cdef double dtw_distance(size_t s_offset,
#                          size_t s_stride,
#                          size_t s_length,
#                          double* S,
#                          size_t t_offset,
#                          size_t t_stride,
#                          size_t t_length,
#                          double* T,
#                          size_t* index) nogil


# cdef int dtw_distance_matches(size_t s_offset,
#                               size_t s_stride,
#                               size_t s_length,
#                               double* S,
#                               size_t t_offset,
#                               size_t t_stride,
#                               size_t t_length,
#                               double* T,
#                               double threshold,
#                               size_t** matches,
#                               size_t* n_matches) nogil except -1


# cdef double scaled_dtw_distance_matches(size_t s_offset,
#                                         size_t s_stride,
#                                         size_t s_length,
#                                         double s_mean,
#                                         double s_std,
#                                         double* S,
#                                         size_t t_offset,
#                                         size_t t_stride,
#                                         size_t t_length,
#                                         double* T,
#                                         double* X_buffer,
#                                         double threshold,
#                                         size_t** matches,
#                                         size_t* n_matches) nogil except -1
