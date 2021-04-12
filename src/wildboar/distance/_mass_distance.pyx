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

# Authors: Isak Samsten

import numpy as np

cimport numpy as np
from libc.math cimport INFINITY, NAN, sqrt
from libc.stdlib cimport free, malloc, realloc

from .._data cimport TSDatabase, ts_database_new
from ._distance cimport DistanceMeasure, ScaledDistanceMeasure, TSCopy, TSView


cdef void simple_z_norm(TSCopy s, double *out) nogil:
    for i in range(s.length):
        out[i] = (s.data[0] - s.mean) / s.std

cdef int movestd(double *T,
                 Py_ssize_t offset,
                 Py_ssize_t stride,
                 Py_ssize_t length,
                 int m,
                 double *out) nogil:
    cdef double current_sum = 0
    cdef double current_sum_square = 0
    cdef double current_value
    cdef Py_ssize_t i
    cdef double *cumsum = <double*> malloc(sizeof(double) * length + 1)
    cdef double *cumsum_square = <double*> malloc(
        sizeof(double) * length + 1)

    cumsum[0] = 0
    cumsum_square[0] = 0
    for i in range(length):
        current_value = T[offset + i * stride]
        current_sum += current_value
        current_sum_square += current_value * current_value
        cumsum[i + 1] = current_sum
        cumsum_square[i + 1] = current_sum_square

    cdef double seq_sum = 0
    cdef double seq_sum_square = 0
    for i in range(length - m + 1):
        seq_sum = cumsum[m + i] - cumsum[i]
        seq_sum_square = cumsum_square[m + i] - cumsum_square[i]
        out[i] = sqrt(seq_sum_square / m - (seq_sum / m) ** 2)

cdef int simple_fft(Py_ssize_t offset,
                    Py_ssize_t stride,
                    Py_ssize_t length,
                    double *T,
                    Py_ssize_t out_offset,
                    Py_ssize_t out_stride,  # out_length = length - 1
                    complex *out) nogil:
    cdef int n = length
    cdef int rank = 1
    cdef fftw_plan p
    p = fftw_plan_many_dft_r2c(rank, &n, 1,
                               T + offset, NULL, stride, offset,
                               out + out_offset, NULL, out_stride, out_offset,
                               FFTW_ESTIMATE)

    fftw_execute(p)
    fftw_destroy_plan(p)
    fftw_cleanup()

cdef int simple_ifft(Py_ssize_t offset,
                     Py_ssize_t stride,
                     Py_ssize_t length,
                     complex *T,
                     Py_ssize_t out_offset,
                     Py_ssize_t out_stride,  # out_length = length + 1
                     double *out) nogil:
    cdef int n = length
    cdef int rank = 1
    cdef fftw_plan p
    p = fftw_plan_many_dft_c2r(rank, &n, 1,
                               T + offset, NULL, stride, offset,
                               out + out_offset, NULL, out_stride, out_offset,
                               FFTW_ESTIMATE)
    fftw_execute(p)

    cdef Py_ssize_t i
    for i in range(length):
        out[offset + i * out_stride] /= length

    fftw_destroy_plan(p)
    fftw_cleanup()

cpdef void test1(np.ndarray data, Py_ssize_t ts):
    cdef TSDatabase td = ts_database_new(data)
    cdef double *reconstruct = <double*> malloc(sizeof(double) * td.n_timestep)
    cdef complex *temp = <complex*> malloc(sizeof(complex) * td.n_timestep)

    cdef Py_ssize_t offset = ts * td.sample_stride + td.dim_stride
    print("offset", offset, td.n_timestep, td.timestep_stride, td.sample_stride)
    simple_fft(offset, td.timestep_stride, td.n_timestep, td.data,
               0, 1, temp)

    simple_ifft(0, 1, td.n_timestep, temp, 0, 1, reconstruct)

    cdef int i
    for i in range(td.n_timestep):
        print("reconstruct", reconstruct[i], temp[i], td.data[i])

cpdef void test() nogil:
    cdef int i
    cdef Py_ssize_t length = 10
    cdef double *T = <double*> malloc(sizeof(double) * length)
    for i in range(length):
        T[i] = i

    T[2] = 1
    T[4] = 2
    T[6] = 3
    T[8] = 10
    cdef double *out2 = <double*> malloc(sizeof(double) * length - 3 + 1)
    movestd(T, 0, 1, length, 3, out2)

    with gil:
        print("testing")
        for i in range(length):
            print("T", T[i])
        for i in range(2):
            print(out2[i])

    cdef complex *out3 = <complex*> malloc(sizeof(complex) * (4 - 1))
    simple_fft(2, 2, 4, T, 0, 1, out3)

    cdef double *out4 = <double*> malloc(sizeof(double) * 4)
    simple_ifft(0, 1, 4, out3, 0, 1, out4)
    with gil:
        for i in range(3):
            print("out3", out3[i])
        for i in range(4):
            print("out4", out4[i])

    cdef int n = 4
    cdef double *in1 = <double*> malloc(sizeof(double) * n)
    cdef complex *out = <complex*> malloc(sizeof(complex) * n)
    cdef double *in2 = <double*> malloc(sizeof(double) * n)

    cdef fftw_plan p, q

    p = fftw_plan_dft_r2c_1d(n, in1, out, FFTW_ESTIMATE)
    in1[0] = 1
    in1[1] = 2
    in1[2] = 3
    in1[3] = 10

    fftw_execute(p)

    q = fftw_plan_dft_c2r_1d(n, out, in2, FFTW_ESTIMATE);
    fftw_execute(q);

    with gil:
        for i in range(n):
            print(in1[i], out[i], in2[i] / 4)
