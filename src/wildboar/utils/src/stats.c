#include "stats.h"

#include <stdlib.h>

void fast_stats_reset(FastStats *fs) {
    fs->mean = 0.0;
    fs->n_samples = 0.0;
    fs->s = 0.0;
    fs->sum = 0.0;
}

void fast_stats_add(FastStats *fs, double weight, double value) {
    double next_m;
    fs->n_samples += weight;
    next_m = fs->mean + (value - fs->mean) / fs->n_samples;
    fs->s += (value - fs->mean) * (value - next_m);
    fs->mean = next_m;
    fs->sum += weight * value;
}

void fast_stats_remove(FastStats *fs, double weight, double value) {
    double old_m;
    if (fs->n_samples == 1.0) {
        fs->n_samples = 0.0;
        fs->mean = 0.0;
        fs->s = 0.0;
    } else {
        old_m = (fs->n_samples * fs->mean - value) / (fs->n_samples - weight);
        fs->s -= (value - fs->mean) * (value - old_m);
        fs->mean = old_m;
        fs->n_samples -= weight;
    }
    fs->sum -= weight * value;
}

double fast_stats_variance(FastStats *self, int sample) {
    double n_samples = sample ? self->n_samples - 1 : self->n_samples;
    return n_samples <= 1 ? 0.0 : self->s / n_samples;
}

double covariance(double *x, double *y, size_t length) {
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double k = 0.0;
    double mean_x = 0.0;
    double mean_y = 0.0;
    double tmp_mean_x, tmp_mean_y, diff_x, diff_y;

    for (int i = 0; i < length; i++) {
        tmp_mean_x = mean_x;
        tmp_mean_y = mean_y;
        diff_x = x[i] - tmp_mean_x;
        diff_y = y[i] - tmp_mean_y;
        k += 1;
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += diff_x * diff_y - diff_x * diff_y / k;
        mean_x = sum_x / k;
        mean_y = sum_y / k;
    }
    return sum_xy / (k - 1);
}

double mean(double *x, size_t length) {
    double v = 0;
    for (int i = 0; i < length; i++) {
        v += x[i];
    }
    return v / length;
}

double variance(double *x, size_t length) {
    if (length == 1) return 0.0;

    double avg = mean(x, length);
    double sum = 0;
    double v = 0;
    for (int i = 0; i < length; i++) {
        v = x[i] - avg;
        sum += v * v;
    }
    return sum / length;
}

double slope(double *x, size_t length) {
    if (length == 1) return 0.0;
    double y_mean = (length + 1) / 2.0;
    double x_mean = 0;
    double mean_diff = 0;
    double mean_y_sqr = 0;
    int j;

    for (int i = 0; i < length; i++) {
        j = i + 1;
        mean_diff += x[i] * j;
        x_mean += x[i];
        mean_y_sqr += j * j;
    }
    mean_diff /= length;
    mean_y_sqr /= length;
    x_mean /= length;
    return (mean_diff - y_mean * x_mean) / (mean_y_sqr - y_mean * y_mean);
}