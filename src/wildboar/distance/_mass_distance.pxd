# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

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

#ctypedef double complex[2]

cdef extern from 'fftw3.h':
    ctypedef struct fftw_plan_struct:
        pass

    ctypedef fftw_plan_struct *fftw_plan

    fftw_plan fftw_plan_r2r_1d(int n,
                               double *in_,
                               double *out_,
                               int kind,
                               unsigned b) nogil

    fftw_plan fftw_plan_dft_r2c_1d(int n,
                                   double *in_,
                                   complex *out_,
                                   unsigned flags) nogil

    fftw_plan fftw_plan_dft_c2r_1d(int n,
                                   complex *in_,
                                   double *out_,
                                   unsigned flags) nogil

    fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n, int howmany,
                                     double *in_, const int *inembed,
                                     int istride, int idist,
                                     complex *out, const int *onembed,
                                     int ostride, int odist,
                                     unsigned flags) nogil

    fftw_plan fftw_plan_many_dft_c2r(int rank, const int *n, int howmany,
                                     complex *in_, const int *inembed,
                                     int istride, int idist,
                                     double *out, const int *onembed,
                                     int ostride, int odist,
                                     unsigned flags) nogil

    void fftw_execute(fftw_plan plan) nogil

    void fftw_destroy_plan(fftw_plan plan) nogil

    void fftw_cleanup() nogil

    cdef int FFTW_ESTIMATE

cdef enum:
    FFTW_FORWARD = -1
    FFTW_BACKWARD = 1
