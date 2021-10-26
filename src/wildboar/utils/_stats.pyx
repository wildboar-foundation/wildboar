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

from libc.stdlib cimport free, malloc

from ._fft cimport _pocketfft


cdef void inc_stats_init(IncStats *self) nogil:
    self._m = 0.0
    self._n_samples = 0.0
    self._s = 0.0
    self._sum = 0.0

cdef void inc_stats_add(IncStats *self, double weight, double value) nogil:
    cdef double next_m
    self._n_samples += weight
    next_m = self._m + (value - self._m) / self._n_samples
    self._s += (value - self._m) * (value - next_m)
    self._m = next_m
    self._sum += weight * value

cdef void inc_stats_remove(IncStats *self, double weight, double value) nogil:
    cdef double old_m
    if self._n_samples == 1.0:
        self._n_samples = 0.0
        self._m = 0.0
        self._s = 0.0
    else:
        old_m = (self._n_samples * self._m - value) / (self._n_samples - weight)
        self._s -= (value - self._m) * (value - old_m)
        self._m = old_m
        self._n_samples -= weight
    self._sum -= weight * value

cdef double inc_stats_n_samples(IncStats *self) nogil:
    return self._n_samples

cdef double inc_stats_sum(IncStats *self) nogil:
    return self._sum

cdef double inc_stats_mean(IncStats *self) nogil:
    return self._m

cdef double inc_stats_variance(IncStats *self, bint sample=True) nogil:
    cdef double n_samples
    if sample:
        n_samples = self._n_samples - 1
    else:
        n_samples = self._n_samples
    return 0.0 if n_samples <= 1 else self._s / n_samples

cdef double mean(Py_ssize_t stride, double *x, Py_ssize_t length) nogil:
    cdef double v
    cdef Py_ssize_t i
    for i in range(length):
        v += x[i * stride]
    return v / length

cdef double variance(Py_ssize_t stride, double *x, Py_ssize_t length) nogil:
    if length == 1:
        return 0.0

    cdef double avg = mean(stride, x, length)
    cdef double sum = 0
    cdef double v
    cdef Py_ssize_t i
    for i in range(length):
        v = x[i * stride] - avg
        sum += v * v
    return sum / length

cdef double slope(Py_ssize_t stride, double *x, Py_ssize_t length) nogil:
    if length == 1:
        return 0.0
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

cdef double covariance(double *x, double *y, Py_ssize_t length) nogil:
    cdef:
        double sum_x = 0.0
        double sum_y = 0.0
        double sum_xy = 0.0
        double k = 0.0
        double mean_x = 0.0
        double mean_y = 0.0
        double tmp_mean_x, tmp_mean_y, diff_x, diff_y
        Py_ssize_t i

    for i in range(length):
        tmp_mean_x = mean_x
        tmp_mean_y = mean_y
        diff_x = x[i] - tmp_mean_x
        diff_y = y[i] - tmp_mean_y
        k += 1
        sum_x += x[i]
        sum_y += y[i]
        sum_xy += diff_x * diff_y - diff_x * diff_y / k
        mean_x = sum_x / k
        mean_y = sum_y / k
    return sum_xy / k



cdef void _auto_correlation(double *x, Py_ssize_t n, double *out, complex *fft) nogil:
    cdef double avg = mean(1, x, n)
    cdef Py_ssize_t fft_length = n * 2 - 1
    cdef Py_ssize_t i
    for i in range(n):
        fft[i] = x[i] - avg
    for i in range(n, fft_length):
        fft[i] = 0.0

    _pocketfft.fft(fft, fft_length, 1.0)
    for i in range(fft_length):
        fft[i] = fft[i] * fft[i].conjugate()

    _pocketfft.ifft(fft, fft_length, 1.0)
    cdef complex first = fft[0]
    for i in range(n):
        out[i] = (fft[i] / first).real

cdef void auto_correlation(double *x, Py_ssize_t n, double *out) nogil:
    cdef Py_ssize_t fft_length = n * 2 - 1
    cdef complex *fft = <complex *> malloc(sizeof(complex) * fft_length)
    _auto_correlation(x, n, out, fft)
    free(fft)
