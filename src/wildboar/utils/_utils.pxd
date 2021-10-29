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


cdef void strided_copy(Py_ssize_t stride, double* f, double* t, Py_ssize_t length) nogil

cdef void argsort(double *values, Py_ssize_t *order, Py_ssize_t length) nogil

cdef int realloc_array(void** a, Py_ssize_t p, Py_ssize_t size, Py_ssize_t *cap)  nogil except -1

cdef int safe_realloc(void** ptr, Py_ssize_t new_size) nogil except -1

cdef np.ndarray to_ndarray_int(Py_ssize_t *arr, Py_ssize_t n)

cdef np.ndarray to_ndarray_double(double *arr, Py_ssize_t n)