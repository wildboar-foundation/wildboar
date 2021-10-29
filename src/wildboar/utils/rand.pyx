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

from libc.math cimport log, sqrt


cdef inline size_t rand_r(size_t *seed) nogil:
    """Returns a pesudo-random number based on the seed."""
    seed[0] = seed[0] * 1103515245 + 12345
    return seed[0] % (<size_t> RAND_R_MAX + 1)


cdef size_t rand_int(size_t min_val, size_t max_val, size_t *seed) nogil:
    """Returns a pseudo-random number in the range [`min_val` `max_val`["""
    if min_val == max_val:
        return min_val
    else:
        return min_val + rand_r(seed) % (max_val - min_val)


cdef double rand_uniform(double low, double high, size_t *random_state) nogil:
    """Generate a random double in the range [`low` `high`[."""
    return ((high - low) * <double> rand_r(random_state) / <double> RAND_R_MAX) + low


cdef double rand_normal(double mu, double sigma, size_t *random_state) nogil:
    cdef double x1, x2, w, _y1
    x1 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
    x2 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
    w = x1 * x1 + x2 * x2
    while w >= 1.0:
        x1 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
        x2 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt((-2.0 * log(w)) / w)
    _y1 = x1 * w
    y2 = x2 * w
    return mu + _y1 * sigma


cdef void shuffle(Py_ssize_t *values, Py_ssize_t length, size_t *seed) nogil:
    cdef Py_ssize_t i, j
    for i in range(length - 1, 0, -1):
        j = rand_int(0, i, seed)
        values[i], values[j] = values[j], values[i]