# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see
# <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten

cimport numpy as np

cdef struct TSView:
    Py_ssize_t index  # the index of the shapelet sample
    Py_ssize_t start  # the start position
    Py_ssize_t length  # the length of the shapelet
    Py_ssize_t dim  # the dimension of the shapelet
    double mean  # the mean of the shapelet
    double std  # the stanard devision
    void *extra

cdef struct TSDatabase:
    Py_ssize_t n_samples  # the number of samples
    Py_ssize_t n_timestep  # the number of timesteps
    Py_ssize_t n_dims

    double *data  # the data
    Py_ssize_t sample_stride  # the stride for samples
    Py_ssize_t timestep_stride  # the `feature` stride
    Py_ssize_t dim_stride  # the dimension stride

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

    cdef int init_ts_view(self, TSDatabase *td, TSView *ts_view, Py_ssize_t index, Py_ssize_t start,
                          Py_ssize_t length, Py_ssize_t dim) nogil

    cdef int init_ts_copy_from_ndarray(self, TSCopy *shapelet, np.ndarray arr, Py_ssize_t dim)

    cdef int init_ts_copy(self, TSCopy *shapelet, TSView *s, TSDatabase *td) nogil

    cdef int ts_copy_sub_matches(self, TSCopy *s, TSDatabase *td, Py_ssize_t t_index, double threshold, Py_ssize_t** matches,
                             double** distances, Py_ssize_t *n_matches) nogil except -1

    cdef double ts_copy_sub_distance(self, TSCopy *s, TSDatabase *td, Py_ssize_t t_index, Py_ssize_t *return_index= *) nogil

    cdef double ts_view_sub_distance(self, TSView *s, TSDatabase *td, Py_ssize_t t_index) nogil

    # Minumum subsequence distance
    cdef void ts_view_sub_distances(self, TSView *s, TSDatabase *td, Py_ssize_t *samples, double *distances,
                                Py_ssize_t n_samples) nogil

    # Distance between s and td[t_index]
    cdef double ts_copy_distance(self, TSCopy *s, TSDatabase *td, Py_ssize_t t_index) nogil

    # Return true if unaligned time series are supported for ts_copy_distance
    cdef bint support_unaligned(self) nogil

cdef class ScaledDistanceMeasure(DistanceMeasure):
    pass

cdef TSDatabase ts_database_new(np.ndarray X)

cdef void ts_view_init(TSView *s) nogil

cdef void ts_view_free(TSView *shapelet_info) nogil

cdef int ts_copy_init(TSCopy *shapelet, Py_ssize_t dim, Py_ssize_t length, double mean, double std) nogil

cdef void ts_copy_free(TSCopy *shapelet) nogil
