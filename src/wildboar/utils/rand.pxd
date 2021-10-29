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

cdef enum:
    RAND_R_MAX = 2147483647

cdef size_t rand_r(size_t *seed) nogil

cdef size_t rand_int(size_t min_val, size_t max_val, size_t *seed) nogil

cdef double rand_uniform(double low, double high, size_t *random_state) nogil

cdef double rand_normal(double mean, double std, size_t *random_state) nogil

cdef void shuffle(Py_ssize_t *values, Py_ssize_t length, size_t *seed) nogil