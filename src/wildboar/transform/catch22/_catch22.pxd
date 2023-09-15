# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef double histogram_mode10(
        const double *x,
        Py_ssize_t length,
        int *bin_count,
        double *bin_edges
) noexcept nogil

cdef double histogram_mode5(
        const double *x,
        Py_ssize_t length,
        int *bin_count,
        double *bin_edges
) noexcept nogil


cdef double histogram_ami_even_2_5(const double *x, Py_ssize_t length) noexcept nogil

cdef double transition_matrix_3ac_sumdiagcov(const double *x, double *ac, Py_ssize_t n) noexcept nogil

cdef double f1ecac(const double *ac, Py_ssize_t n) noexcept nogil

cdef double first_min(double *ac, Py_ssize_t n) noexcept nogil

cdef double trev_1_num(const double *x, Py_ssize_t n) noexcept nogil

cdef double local_mean_std(const double *x, Py_ssize_t n, Py_ssize_t lag) noexcept nogil

cdef double hrv_classic_pnn(const double *x, Py_ssize_t n, double pnn) noexcept nogil

cdef double above_mean_stretch(const double *x, Py_ssize_t n) noexcept nogil

cdef double below_diff_stretch(const double *x, Py_ssize_t n) noexcept nogil

cdef double local_mean_tauresrat(const double *x, double *ac, Py_ssize_t n, Py_ssize_t lag) noexcept nogil

cdef double periodicity_wang_th0_01(const double *x, int length) noexcept nogil

cdef double embed2_dist_tau_d_expfit_meandiff(const double *x, double *ac, int length) noexcept nogil

cdef double auto_mutual_info_stats_gaussian_fmmi(const double *x, int length, int tau) nogil 

cdef double outlier_include_np_mdrmd(const double *x, int length, int sign, double inc) noexcept nogil

cdef double summaries_welch_rect(
    const double *x, int length, int what, double *S, double *f, int n_welch
) noexcept nogil

cdef double motif_three_quantile_hh(const double *x, int length) noexcept nogil

cdef double fluct_anal_2_50_1_logi_prop_r1(const double *x, int length, int lag, int how) noexcept nogil
