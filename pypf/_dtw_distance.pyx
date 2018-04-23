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

from libc.stdlib cimport malloc
from libc.stdlib cimport free

cdef void init_circular(CircularArray* c, size_t capacity) nogil:
    c[0].capacity = capacity
    c[0].size = 0
    c[0].queue = <size_t*>malloc(sizeof(size_t) * capacity)
    c[0].front = 0
    c[0].back = capacity - 1

cdef void circular_reset(CircularArray* c) nogil:
    c[0].size = 0
    c[0].front = 0
    c[0].back = c[0].capacity - 1
    # TODO: is memset(...) required?


cdef void destroy_cicular(CircularArray* c) nogil:
    free(c[0].queue)


cdef void cirular_push_back(CircularArray* c, size_t v) nogil:
    c[0].queue[c[0].back] = v
    c[0].back =- 1
    if c[0].back < 0:
        c[0].back = c[0].capacity - 1

    c[0].size += 1


cdef void circular_pop_front(CircularArray* c) nogil:
    c[0].front -= 1
    if c[0].front < 0:
        c[0].front = c[0].capacity - 1
    c[0].size -= 1

cdef void cicular_pop_back(CircularArray* c) nogil:
    c[0].back = (c[0].back + 1) % c[0].capacity
    c[0].size -= 1

cdef size_t circular_front(CircularArray* c) nogil:
    cdef size_t tmp = c[0].front - 1
    if tmp < 0:
        tmp = c[0].capacity - 1
    return c[0].queue[tmp]

cdef size_t circular_back(CircularArray* c) nogil:
    cdef size_t tmp = (c[0].back + 1) % c[0].capacity
    return c[0].queue[tmp]

cdef size_t circular_size(CircularArray* c) nogil:
    return c[0].size


cdef void find_min_max(size_t offset, size_t stride, size_t length,
                       double* T, size_t r, double* lower, double* upper,
                       CircularArray* du, CircularArray* dl) nogil:
    cdef size_t i
    cdef size_t k

    circular_reset(du)

    for i in range(1, length):
        if i > r:
            k = i - r - 1
            lower[k] = T[offset + stride * circular_front(dl)]
            upper[k] = T[offset + stride * circular_front(du)]


        if T[offset + stride * i] > T[offset + stride * (i - 1)]:
            circular_pop_back(du)

            while (cirular_size(du) > 0 and
                   T[i] > T[offset + stride * circular_back(du)]):
                circular_pop_back(du)
        else:
            circular_pop_back(dl)

            while (circular_size(dl) > 0 and
                   T[i] > T[offset + stride * circular_back(dl)]):
                circular_pop_back(dl)

        circular_push_back(du, i)
        circular_push_back(dl, i)

        if i == 2 * r + 1 + circular_front(du):
            circular_pop_front(du)

        elif i == 2 * r + 1 + circular_back(dl):
            circular_pop_front(dl)

    for i in range(length, length + r + 1):
        pass # WIP
        
                
    
