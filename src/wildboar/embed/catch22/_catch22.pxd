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

cdef double _histogram_mode(
    double *x,
    Py_ssize_t length,
    size_t *bin_count,
    double *bin_edges,
    size_t n_bins,
) nogil

cdef double histogram_mode10(
        double *x,
        Py_ssize_t length,
        size_t *bin_count,
        double *bin_edges
) nogil

cdef double histogram_mode5(
        double *x,
        Py_ssize_t length,
        size_t *bin_count,
        double *bin_edges
) nogil

cdef double _histogram_ami_even(
    double *x,
    Py_ssize_t length,
    size_t tau,
    size_t n_bins,
) nogil

cdef double histogram_ami_even_2_5(double *x, Py_ssize_t length) nogil

cdef double transition_matrix_3ac_sumdiagcov(double *x, double *ac, Py_ssize_t n) nogil

cdef double f1ecac(double *ac, Py_ssize_t n) nogil

cdef double first_min(double *ac, Py_ssize_t n) nogil

cdef double trev_1_num(Py_ssize_t stride, double *x, Py_ssize_t n) nogil

cdef double local_mean_std(Py_ssize_t stride, double *x, Py_ssize_t n, Py_ssize_t lag) nogil

cdef double hrv_classic_pnn(Py_ssize_t stride, double *x, Py_ssize_t n, double pnn) nogil

cdef double above_mean_stretch(Py_ssize_t stride, double *x, Py_ssize_t n) nogil

cdef double local_mean_tauresrat(double *x, double *ac, Py_ssize_t n, Py_ssize_t lag) nogil

cdef double periodicity_wang_th0_01(double *x, size_t length) nogil

cdef double embed2_dist_tau_d_expfit_meandiff(double *x, double *ac, size_t length) nogil

cdef double auto_mutual_info_stats_gaussian_fmmi(double *x, size_t length, size_t tau) nogil 

cdef double outlier_include_np_mdrmd(double *x, size_t length, int sign, double inc) nogil

cdef double summaries_welch_rect(
    double *x, size_t length, int what, double *S, double *f, size_t n_welch
) nogil