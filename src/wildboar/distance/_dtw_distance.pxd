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

cdef struct DtwExtra:
    double *lower
    double *upper

cdef struct Deque:
    Py_ssize_t *queue
    int size
    int capacity
    int front
    int back

cdef void deque_init(Deque *c, Py_ssize_t capacity) nogil

cdef void deque_destroy(Deque *c) nogil

cdef void deque_push_back(Deque *c, Py_ssize_t v) nogil

cdef void deque_pop_front(Deque *c) nogil

cdef void deque_pop_back(Deque *c) nogil

cdef Py_ssize_t deque_front(Deque *c) nogil

cdef Py_ssize_t deque_back(Deque *c) nogil

cdef Py_ssize_t deque_size(Deque *c) nogil

cdef bint deque_empty(Deque *c) nogil

cdef void _dtw_align(double[:] A, double[:] B, Py_ssize_t warp_width, double[:,:] warp) nogil

cdef double _dtw(Py_ssize_t x_offset, Py_ssize_t x_stride, Py_ssize_t x_length, double *X, double x_mean, double x_std,
                 Py_ssize_t y_offset, Py_ssize_t y_stride, Py_ssize_t y_length, double *Y, double y_mean, double y_std,
                 Py_ssize_t r, double *cost, double *cost_prev) nogil