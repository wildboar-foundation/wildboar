# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten

cdef class RollingVariance:
    cdef double _n_samples
    cdef double _m
    cdef double _s
    cdef double _sum

    cdef void _reset(self) nogil

    cdef void _add(self, double weight, double value) nogil

    cdef void _remove(self, double weight, double value) nogil

    cdef double _mean(self) nogil

    cdef double _variance(self) nogil

cdef enum:
    RAND_R_MAX = 2147483647

cdef void print_c_array_d(object name, double *arr, size_t length)

cdef void print_c_array_i(object name, size_t *arr, size_t length)

cdef size_t label_distribution(const size_t *samples, double *sample_weights, size_t start, size_t end,
                               const size_t *labels, size_t labels_stride, size_t n_classes,
                               double*n_weighted_samples, double*dist) nogil

cdef void argsort(double *values, size_t *order, size_t length) nogil

cdef size_t rand_r(size_t *seed) nogil

cdef size_t rand_int(size_t min_val, size_t max_val, size_t *seed) nogil

cdef double rand_uniform(double low, double high, size_t *random_state) nogil

cdef int realloc_array(void** a, size_t p, size_t size, size_t *cap)  nogil except -1

cdef int safe_realloc(void** ptr, size_t new_size) nogil except -1
