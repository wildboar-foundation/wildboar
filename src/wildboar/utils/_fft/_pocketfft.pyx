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

cdef extern from "pocketfft.h":
    ctypedef struct rfft_plan:
       pass

    ctypedef struct cfft_plan:
       pass

    cdef rfft_plan make_rfft_plan (size_t length) nogil
    cdef void destroy_rfft_plan (rfft_plan plan) nogil
    cdef int rfft_backward(rfft_plan plan, double *c, double fct) nogil
    cdef int rfft_forward(rfft_plan plan, double *c, double fct) nogil
    cdef size_t rfft_length(rfft_plan plan) nogil

    cdef cfft_plan make_cfft_plan(size_t length) nogil
    cdef void destroy_cfft_plan(cfft_plan plan) nogil
    cdef int cfft_backward(cfft_plan plan, double *c, double fct) nogil
    cdef int cfft_forward(cfft_plan plan, double *c, double fct) nogil
    cdef size_t cfft_length(cfft_plan plan) nogil

cdef void fft(complex *x, Py_ssize_t n, double fct) nogil:
    cdef cfft_plan fft_plan = make_cfft_plan(n)
    cfft_forward(fft_plan, <double *> x, fct)
    destroy_cfft_plan(fft_plan)

cdef void ifft(complex *x, Py_ssize_t n, double fct) nogil:
    cdef cfft_plan fft_plan = make_cfft_plan(n)
    cfft_backward(fft_plan, <double *> x, fct)
    destroy_cfft_plan(fft_plan)

cdef void rfft(double *x, Py_ssize_t n, double fct) nogil:
    cdef rfft_plan fft_plan = make_rfft_plan(n)
    rfft_forward(fft_plan, x, fct)
    destroy_rfft_plan(fft_plan)

cdef void irfft(double *x, Py_ssize_t n, double fct) nogil:
    cdef rfft_plan fft_plan = make_rfft_plan(n)
    rfft_backward(fft_plan, x, fct)
    destroy_rfft_plan(fft_plan)