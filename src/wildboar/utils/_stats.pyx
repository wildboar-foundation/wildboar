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

cdef double mean(Py_ssize_t stride, double *x, Py_ssize_t length) nogil:
    cdef double v
    cdef Py_ssize_t i
    for i in range(length):
        v += x[i * stride]
    return v / length

cdef double variance(Py_ssize_t stride, double *x, Py_ssize_t length) nogil:
    cdef double avg = mean(stride, x, length)
    cdef double sum = 0
    cdef double v
    cdef Py_ssize_t i
    for i in range(length):
        v = x[i * stride] - avg
        sum += v * v
    return sum / length

cdef double slope(Py_ssize_t stride, double *x, Py_ssize_t length) nogil:
    cdef double y_mean = (length + 1) / 2.0
    cdef double x_mean = 0
    cdef double mean_diff = 0
    cdef double mean_y_sqr = 0
    cdef Py_ssize_t i, j

    for i in range(length):
        j = i + 1
        mean_diff += x[stride * i] * j
        x_mean += x[stride * i]
        mean_y_sqr += j * j
    mean_diff /= length
    mean_y_sqr /= length
    x_mean /= length
    return (mean_diff - y_mean * x_mean) / (mean_y_sqr - y_mean ** 2)