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

cdef double histogram_mode(
    Py_ssize_t stride,
    double *x,
    Py_ssize_t length,
    Py_ssize_t *bin_count,
    double *bin_edges,
    Py_ssize_t n_bins,
) nogil

cdef double f1ecac(double *ac, Py_ssize_t n) nogil

cdef double first_min(double *ac, Py_ssize_t n) nogil

cdef double trev_1_num(Py_ssize_t stride, double *x, Py_ssize_t n) nogil

cdef double local_mean_std(Py_ssize_t stride, double *x, Py_ssize_t n, Py_ssize_t lag) nogil

cdef double hrv_classic_pnn(Py_ssize_t stride, double *x, Py_ssize_t n, double pnn) nogil

cdef double above_mean_stretch(Py_ssize_t stride, double *x, Py_ssize_t n) nogil

cdef double transition_matrix_3ac_sumdiagcov(double *x, double *ac, Py_ssize_t n) nogil

cdef double local_mean_tauresrat(double *x, double *ac, Py_ssize_t n, Py_ssize_t lag) nogil