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
#
# Authors: Isak Samsten


cdef double scaled_euclidean_distance(
    Py_ssize_t s_offset,
    Py_ssize_t s_stride,
    Py_ssize_t s_length,
    double s_mean,
    double s_std,
    double *S,
    Py_ssize_t t_offset,
    Py_ssize_t t_stride,
    Py_ssize_t t_length,
    double *T,
    double *X_buffer,
    Py_ssize_t *index,
) nogil


cdef double euclidean_distance(
    Py_ssize_t s_offset,
    Py_ssize_t s_stride,
    Py_ssize_t s_length,
    double *S,
    Py_ssize_t t_offset,
    Py_ssize_t t_stride,
    Py_ssize_t t_length,
    double *T,
    Py_ssize_t *index,
) nogil
