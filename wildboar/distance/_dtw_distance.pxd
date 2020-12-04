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

cdef struct DtwExtra:
    double *lower
    double *upper

cdef struct Deque:
    size_t *queue
    int size
    int capacity
    int front
    int back

cdef void deque_init(Deque *c, size_t capacity) nogil

cdef void deque_destroy(Deque *c) nogil

cdef void deque_push_back(Deque *c, size_t v) nogil

cdef void deque_pop_front(Deque *c) nogil

cdef void deque_pop_back(Deque *c) nogil

cdef size_t deque_front(Deque *c) nogil

cdef size_t deque_back(Deque *c) nogil

cdef size_t deque_size(Deque *c) nogil

cdef bint deque_empty(Deque *c) nogil
