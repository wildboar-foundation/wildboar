/**
 * This file is part of wildboar.
 *
 * Author: Isak Samsten
 */
#ifndef CATCH22_H
#define CATCH22_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

double histogram_mode(const double *x, int length, int *hist_counts, double *bin_edges,
                      int n_bins);

double histogram_ami_even(const double *x, int length, int tau, int n_bins);

double transition_matrix_ac_sumdiagcov(const double *x, double *ac, int length, int n_groups);

double periodicity_wang_th0_01(const double *x, int length);

double embed2_dist_tau_d_expfit_meandiff(const double *x, double *ac, int length);

double auto_mutual_info_stats_gaussian_fmmi(const double *x, int length, int tau);

double outlier_include_np_mdrmd(const double *x, int length, int sign, double inc);

double summaries_welch_rect(const double *x, int length, int what, double *S, double *f,
                            int n_welch);

double motif_three_quantile_hh(const double *x, int size);

double fluct_anal_2_50_1_logi_prop_r1(const double *y, int size, int lag, int how);

#endif