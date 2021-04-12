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

cimport numpy as np


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

cdef void print_c_array_d(object name, double *arr, Py_ssize_t length)

cdef void print_c_array_i(object name, Py_ssize_t *arr, Py_ssize_t length)

cdef void argsort(double *values, Py_ssize_t *order, Py_ssize_t length) nogil

cdef size_t rand_r(size_t *seed) nogil

cdef size_t rand_int(size_t min_val, size_t max_val, size_t *seed) nogil

cdef double rand_uniform(double low, double high, size_t *random_state) nogil

cdef double rand_normal(double mean, double std, size_t *random_state) nogil

cdef int realloc_array(void** a, Py_ssize_t p, Py_ssize_t size, Py_ssize_t *cap)  nogil except -1

cdef int safe_realloc(void** ptr, Py_ssize_t new_size) nogil except -1

cdef void fast_mean_std(Py_ssize_t offset, Py_ssize_t stride, Py_ssize_t length, double* data, double *mean, double* std) nogil

cpdef check_array_fast(np.ndarray x, bint ensure_2d=*, bint allow_nd=*, bint c_order=*)