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

cdef struct IncStats:
    double _n_samples
    double _m
    double _s
    double _sum

cdef void inc_stats_init(IncStats *inc_stats) nogil

cdef void inc_stats_add(IncStats *inc_stats, double weight, double value) nogil

cdef void inc_stats_remove(IncStats *inc_stats, double weight, double value) nogil

cdef double inc_stats_mean(IncStats *inc_stats) nogil

cdef double inc_stats_variance(IncStats *inc_stats, bint sample=*) nogil

cdef double inc_stats_n_samples(IncStats *inc_stats) nogil

cdef double inc_stats_sum(IncStats *inc_stats) nogil

cdef double mean(double *x, Py_ssize_t length) nogil

cdef double variance(double *x, Py_ssize_t length) nogil

cdef double slope(double *x, Py_ssize_t length) nogil

cdef double covariance(double *x, double *y, Py_ssize_t length) nogil

cdef void auto_correlation(double *x, Py_ssize_t n, double *out) nogil

cdef void _auto_correlation(double *x, Py_ssize_t n, double *out, complex *fft) nogil

cdef Py_ssize_t next_power_of_2(Py_ssize_t n) nogil

cdef int welch(
    double *x, 
    Py_ssize_t size, 
    int NFFT, 
    double Fs, 
    double *window, 
    int windowWidth,
    double *Pxx, 
    double *f,
) nogil


cdef void fast_mean_std(
    double* data,
    Py_ssize_t length,
    double *mean,
    double* std,
) nogil