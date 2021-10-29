#ifndef STATS_H
#define STATS_H

#include <stdlib.h>

typedef struct _FastStats {
    double n_samples;
    double mean;
    double s;
    double sum;
} FastStats;

void fast_stats_reset(FastStats *fs);

void fast_stats_add(FastStats *fs, double weight, double value);

void fast_stats_remove(FastStats *fs, double weight, double value);

double fast_stats_variance(FastStats *self, int sample);

double mean(double *x, int length);

double variance(double *x, int length);

double slope(double *x, int length);

double covariance(double *x, double *y, int length);

#endif