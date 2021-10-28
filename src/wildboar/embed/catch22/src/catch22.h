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
#endif