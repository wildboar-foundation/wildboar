#ifndef CATCH22_H
#define CATCH22_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

double histogram_mode(double *x, size_t length, size_t *hist_counts, double *bin_edges,
                      size_t n_bins);

double histogram_ami_even(double *x, size_t length, size_t tau, size_t n_bins);

double transition_matrix_ac_sumdiagcov(double *x, double *ac, size_t length,
                                       size_t n_groups);

double periodicity_wang_th0_01(double *x, size_t length);

double embed2_dist_tau_d_expfit_meandiff(double *x, double *ac, size_t length);

double auto_mutual_info_stats_gaussian_fmmi(double *x, size_t length, size_t tau);

double outlier_include_np_mdrmd(double *x, size_t length, int sign, double inc);

double summaries_welch_rect(double *x, size_t length, int what, double *S, double *f,
                            size_t n_welch);
#endif