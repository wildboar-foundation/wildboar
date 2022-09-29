/**
 * This file is part of wildboar.
 *
 * This is a modified implementation of a subset of catch22.
 *
 * Original author: Carl Henning Lubba
 *
 * Author: Isak Samsten
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double _covariance(double *x, double *y, int length) {
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

double _mean(double *x, int length) {
    double v = 0;
    for (int i = 0; i < length; i++) {
        v += x[i];
    }
    return v / length;
}

static int _double_compare(const void *a, const void *b) {
    if (*(double *)a < *(double *)b) {
        return -1;
    } else if (*(double *)a > *(double *)b) {
        return 1;
    } else {
        return 0;
    }
}

void _sort_double(double *x, int length) {
    qsort(x, length, sizeof(double), _double_compare);
}

double _quantile(double *x_sorted, int length, double quant) {
    double quant_idx, q, value;
    int idx_left, idx_right;

    q = 0.5 / length;
    if (quant < q) {
        value = x_sorted[0];
        return value;
    } else if (quant > (1 - q)) {
        value = x_sorted[length - 1];
        return value;
    }

    quant_idx = length * quant - 0.5;
    idx_left = (int)floor(quant_idx);
    idx_right = (int)ceil(quant_idx);
    value = x_sorted[idx_left] + (quant_idx - idx_left) *
                                     (x_sorted[idx_right] - x_sorted[idx_left]) /
                                     (idx_right - idx_left);
    return value;
}

void histcount(double *x, int size, int n_bins, int *bin_count, double *bin_edges) {
    double min_val = INFINITY;
    double max_val = -INFINITY;
    memset(bin_count, 0, sizeof(int) * n_bins);

    for (int i = 0; i < size; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
        }
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    double bin_step = (max_val - min_val) / n_bins;
    for (int i = 0; i < size; i++) {
        int bin = (int)((x[i] - min_val) / bin_step);
        if (bin < 0) {
            bin = 0;
        }
        if (bin >= n_bins) {
            bin = n_bins - 1;
        }
        bin_count[bin] += 1;
    }

    for (int i = 0; i < n_bins + 1; i++) {
        bin_edges[i] = i * bin_step + min_val;
    }
}

double histogram_mode(double *x, int length, int *hist_counts, double *bin_edges,
                      int n_bins) {
    histcount(x, length, n_bins, hist_counts, bin_edges);

    int max_count = 0;
    int n_max_value = 1;
    double out = 0;
    for (int i = 0; i < n_bins; i++) {
        if (hist_counts[i] > max_count) {
            max_count = hist_counts[i];
            n_max_value = 1;
            out = (bin_edges[i] + bin_edges[i + 1]) * 0.5;
        } else if (hist_counts[i] == max_count) {
            n_max_value += 1;
            out += (bin_edges[i] + bin_edges[i + 1]) * 0.5;
        }
    }
    out /= n_max_value;
    return out;
}

void histbinassign(double *x, const int length, double *bin_edges, int n_edges,
                   int *bin) {
    for (int i = 0; i < length; i++) {
        bin[i] = 0;
        for (int j = 0; j < n_edges; j++) {
            if (x[i] < bin_edges[j]) {
                bin[i] = j;
                break;
            }
        }
    }
}
void histcount_edges(double *x, int length, double *bin_edges, int n_edges,
                     int *histcounts) {
    memset(histcounts, 0, sizeof(int) * n_edges);
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < n_edges; j++) {
            if (x[i] <= bin_edges[j]) {
                histcounts[j] += 1;
                break;
            }
        }
    }
}

double histogram_ami_even(double *x, int length, int tau, int n_bins) {
    double *x_lag = malloc((length - tau) * sizeof(double));
    double *x_pad = malloc((length - tau) * sizeof(double));

    for (int i = 0; i < length - tau; i++) {
        x_lag[i] = x[i];
        x_pad[i] = x[i + tau];
    }

    double min_val = INFINITY;
    double max_val = -INFINITY;

    for (int i = 0; i < length; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
        }
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    double *bin_edges = (double *)malloc(sizeof(double) * n_bins + 1);
    double step_value = (max_val - min_val + 0.2) / n_bins;
    for (int i = 0; i < n_bins + 1; i++) {
        bin_edges[i] = min_val + step_value * i - 0.1;
    }

    int *lag_bins = (int *)malloc(sizeof(int) * length - tau);
    int *pad_bins = (int *)malloc(sizeof(int) * length - tau);
    histbinassign(x_lag, length - tau, bin_edges, n_bins + 1, lag_bins);
    histbinassign(x_pad, length - tau, bin_edges, n_bins + 1, pad_bins);

    double *merge_bins = malloc((length - tau) * sizeof(double));
    double *merge_bin_edges = malloc(sizeof(double) * (n_bins + 1) * (n_bins + 1));
    for (int i = 0; i < length - tau; i++) {
        merge_bins[i] = (lag_bins[i] - 1) * (n_bins + 1) + pad_bins[i];
    }

    for (int i = 0; i < (n_bins + 1) * (n_bins + 1); i++) {
        merge_bin_edges[i] = i + 1;
    }

    int *joint_hist_count = (int *)malloc(sizeof(int) * (n_bins + 1) * (n_bins + 1));
    histcount_edges(merge_bins, length - tau, merge_bin_edges,
                    (n_bins + 1) * (n_bins + 1), joint_hist_count);

    double *pij = (double *)malloc(sizeof(double) * n_bins * n_bins);
    int sumBins = 0;
    for (int i = 0; i < n_bins; i++) {
        for (int j = 0; j < n_bins; j++) {
            pij[j + i * n_bins] = joint_hist_count[i * (n_bins + 1) + j];
            sumBins += pij[j + i * n_bins];
        }
    }

    for (int i = 0; i < n_bins; i++) {
        for (int j = 0; j < n_bins; j++) {
            pij[j + i * n_bins] /= sumBins;
        }
    }

    double *pi = (double *)malloc(sizeof(double) * n_bins);
    double *pj = (double *)malloc(sizeof(double) * n_bins);
    memset(pi, 0, sizeof(double) * n_bins);
    memset(pj, 0, sizeof(double) * n_bins);
    for (int i = 0; i < n_bins; i++) {
        for (int j = 0; j < n_bins; j++) {
            pi[i] += pij[i * n_bins + j];
            pj[j] += pij[i * n_bins + j];
        }
    }

    double ami = 0;
    for (int i = 0; i < n_bins; i++) {
        for (int j = 0; j < n_bins; j++) {
            if (pij[i * n_bins + j] > 0) {
                ami += pij[i * n_bins + j] * log(pij[i * n_bins + j] / (pj[j] * pi[i]));
            }
        }
    }

    free(lag_bins);
    free(pad_bins);
    free(joint_hist_count);
    free(bin_edges);

    free(x_lag);
    free(x_pad);
    free(merge_bins);
    free(merge_bin_edges);

    free(pi);
    free(pj);
    free(pij);

    return ami;
}

int _is_constant(double *x, int length) {
    for (int i = 0; i < length; i++) {
        if (x[i] != x[0]) {
            return 0;
        }
    }
    return 1;
}

int _co_firstzero(double *ac, const int length, const int max_tau) {
    int ind = 0;
    while (ac[ind] > 0 && ind < max_tau) {
        ind += 1;
    }
    return ind;
}

void _sb_coarsegrain(double *x, int length, const int num_groups, int *labels) {
    double *x_sorted = malloc(length * sizeof(double));
    memcpy(x_sorted, x, length * sizeof(double));
    _sort_double(x_sorted, length);

    double *quantile_threshold =
        malloc((num_groups + 1) * 2 * sizeof(quantile_threshold));
    double step_size = 1.0 / num_groups;
    double step_value = 0;
    for (int i = 0; i < num_groups + 1; i++) {
        quantile_threshold[i] = _quantile(x_sorted, length, step_value);
        step_value += step_size;
    }

    quantile_threshold[0] -= 1;
    for (int i = 0; i < num_groups; i++) {
        for (int j = 0; j < length; j++) {
            if (x[j] > quantile_threshold[i] && x[j] <= quantile_threshold[i + 1]) {
                labels[j] = i + 1;
            }
        }
    }

    free(x_sorted);
    free(quantile_threshold);
}

double transition_matrix_ac_sumdiagcov(double *x, double *ac, int length,
                                       int n_groups) {
    if (_is_constant(x, length)) {
        return NAN;
    }

    n_groups = 3;  // TODO: generalize
    int tau = _co_firstzero(ac, length, length);

    int n_down = (length - 1) / tau + 1;
    double *x_down = malloc(n_down * sizeof(double));
    for (int i = 0; i < n_down; i++) {
        x_down[i] = x[i * tau];
    }

    int *labels = malloc(n_down * sizeof(int));
    _sb_coarsegrain(x_down, n_down, n_groups, labels);

    double T[3][3];
    for (int i = 0; i < n_groups; i++) {
        for (int j = 0; j < n_groups; j++) {
            T[i][j] = 0;
        }
    }

    for (int j = 0; j < n_down - 1; j++) {
        T[labels[j] - 1][labels[j + 1] - 1] += 1;
    }

    for (int i = 0; i < n_groups; i++) {
        for (int j = 0; j < n_groups; j++) {
            T[i][j] /= (n_down - 1);
        }
    }

    double column1[3] = {0};
    double column2[3] = {0};
    double column3[3] = {0};

    for (int i = 0; i < n_groups; i++) {
        column1[i] = T[i][0];
        column2[i] = T[i][1];
        column3[i] = T[i][2];
    }

    double *columns[3];
    columns[0] = &(column1[0]);
    columns[1] = &(column2[0]);
    columns[2] = &(column3[0]);

    double COV[3][3];
    double covTemp = 0;
    for (int i = 0; i < n_groups; i++) {
        for (int j = i; j < n_groups; j++) {
            covTemp = _covariance(columns[i], columns[j], 3);
            COV[i][j] = covTemp;
            COV[j][i] = covTemp;
        }
    }

    double sumdiagcov = 0;
    for (int i = 0; i < n_groups; i++) {
        sumdiagcov += COV[i][i];
    }

    free(x_down);
    free(labels);

    return sumdiagcov;
}

#define nCoeffs 3
#define nPoints 4

#define pieces 2
#define nBreaks 3
#define deg 3
#define nSpline 4
#define piecesExt 8  // 3 * deg - 1

void matrix_multiply(const int sizeA1, const int sizeA2, const double *A,
                     const int sizeB1, const int sizeB2, const double *B, double *C) {
    if (sizeA2 != sizeB1) {
        return;
    }
    for (int i = 0; i < sizeA1; i++) {
        for (int j = 0; j < sizeB2; j++) {
            C[i * sizeB2 + j] = 0;
            for (int k = 0; k < sizeB1; k++) {
                C[i * sizeB2 + j] += A[i * sizeA2 + k] * B[k * sizeB2 + j];
            }
        }
    }
}

void matrix_times_vector(const int sizeA1, const int sizeA2, const double *A,
                         const int sizeb, const double *b, double *c) {
    if (sizeA2 != sizeb) {
        return;
    }

    for (int i = 0; i < sizeA1; i++) {
        c[i] = 0;
        for (int k = 0; k < sizeb; k++) {
            c[i] += A[i * sizeA2 + k] * b[k];
        }
    }
}

void gauss_elimination(int size, double *A, double *b, double *x) {
    double factor;
    double *AElim[nSpline + 1];
    for (int i = 0; i < size; i++) {
        AElim[i] = (double *)malloc(size * sizeof(double));
    }
    double *bElim = malloc(size * sizeof(double));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            AElim[i][j] = A[i * size + j];
        }
        bElim[i] = b[i];
    }

    // go through columns in outer loop
    for (int i = 0; i < size; i++) {
        // go through rows to eliminate
        for (int j = i + 1; j < size; j++) {
            factor = AElim[j][i] / AElim[i][i];

            // subtract in vector
            bElim[j] = bElim[j] - factor * bElim[i];

            // go through entries of this row
            for (int k = i; k < size; k++) {
                AElim[j][k] = AElim[j][k] - factor * AElim[i][k];
            }
        }
    }

    double bMinusATemp;
    for (int i = size - 1; i >= 0; i--) {
        bMinusATemp = bElim[i];
        for (int j = i + 1; j < size; j++) {
            bMinusATemp -= x[j] * AElim[i][j];
        }

        x[i] = bMinusATemp / AElim[i][i];
    }
    for (int i = 0; i < size; i++) free(AElim[i]);
    free(bElim);
}

void lsqsolve_sub(const int sizeA1, const int sizeA2, const double *A, const int sizeb,
                  const double *b, double *x) {
    double *AT = malloc(sizeA2 * sizeA1 * sizeof(double));
    double *ATA = malloc(sizeA2 * sizeA2 * sizeof(double));
    double *ATb = malloc(sizeA2 * sizeof(double));

    for (int i = 0; i < sizeA1; i++) {
        for (int j = 0; j < sizeA2; j++) {
            AT[j * sizeA1 + i] = A[i * sizeA2 + j];
        }
    }

    matrix_multiply(sizeA2, sizeA1, AT, sizeA1, sizeA2, A, ATA);
    matrix_times_vector(sizeA2, sizeA1, AT, sizeA1, b, ATb);
    gauss_elimination(sizeA2, ATA, ATb, x);

    free(AT);
    free(ATA);
    free(ATb);
}

int iLimit(int x, int lim) { return x < lim ? x : lim; }

void icumsum(const int a[], const int size, int b[]) {
    b[0] = a[0];
    for (int i = 1; i < size; i++) {
        b[i] = a[i] + b[i - 1];
    }
}

double cov_mean(const double x[], const double y[], const int size) {
    double covariance = 0;

    for (int i = 0; i < size; i++) {
        covariance += x[i] * y[i];
    }

    return covariance / size;
}

double autocov_lag(const double x[], const int size, const int lag) {
    return cov_mean(x, &(x[lag]), size - lag);
}

int splinefit(const double *y, const int size, double *yOut) {
    int breaks[nBreaks];
    breaks[0] = 0;
    breaks[1] = (int)floor((double)size / 2.0) - 1;
    breaks[2] = size - 1;

    // -- splinebase

    // spacing
    int h0[2];
    h0[0] = breaks[1] - breaks[0];
    h0[1] = breaks[2] - breaks[1];

    // const int pieces = 2;

    // repeat spacing
    int hCopy[4];
    hCopy[0] = h0[0], hCopy[1] = h0[1], hCopy[2] = h0[0], hCopy[3] = h0[1];

    // to the left
    int hl[deg];
    hl[0] = hCopy[deg - 0];
    hl[1] = hCopy[deg - 1];
    hl[2] = hCopy[deg - 2];

    int hlCS[deg];  // cumulative sum
    icumsum(hl, deg, hlCS);

    int bl[deg];
    for (int i = 0; i < deg; i++) {
        bl[i] = breaks[0] - hlCS[i];
    }

    // to the left
    int hr[deg];
    hr[0] = hCopy[0];
    hr[1] = hCopy[1];
    hr[2] = hCopy[2];

    int hrCS[deg];  // cumulative sum
    icumsum(hr, deg, hrCS);

    int br[deg];
    for (int i = 0; i < deg; i++) {
        br[i] = breaks[2] + hrCS[i];
    }

    // add breaks
    int breaksExt[3 * deg];
    for (int i = 0; i < deg; i++) {
        breaksExt[i] = bl[deg - 1 - i];
        breaksExt[i + 3] = breaks[i];
        breaksExt[i + 6] = br[i];
    }
    int hExt[3 * deg - 1];
    for (int i = 0; i < deg * 3 - 1; i++) {
        hExt[i] = breaksExt[i + 1] - breaksExt[i];
    }
    // const int piecesExt = 3*deg-1;

    // initialise polynomial coefficients
    double coefs[nSpline * piecesExt][nSpline + 1];
    for (int i = 0; i < nSpline * piecesExt; i++) {
        for (int j = 0; j < nSpline; j++) {
            coefs[i][j] = 0;
        }
    }
    for (int i = 0; i < nSpline * piecesExt; i = i + nSpline) {
        coefs[i][0] = 1;
    }

    // expand h using the index matrix ii
    int ii[deg + 1][piecesExt];
    for (int i = 0; i < piecesExt; i++) {
        ii[0][i] = iLimit(0 + i, piecesExt - 1);
        ii[1][i] = iLimit(1 + i, piecesExt - 1);
        ii[2][i] = iLimit(2 + i, piecesExt - 1);
        ii[3][i] = iLimit(3 + i, piecesExt - 1);
    }

    // expanded h
    double H[(deg + 1) * piecesExt];
    int iiFlat;
    for (int i = 0; i < nSpline * piecesExt; i++) {
        iiFlat = ii[i % nSpline][i / nSpline];
        H[i] = hExt[iiFlat];
    }

    // recursive generation of B-splines
    double Q[nSpline][piecesExt];
    for (int k = 1; k < nSpline; k++) {
        // antiderivatives of splines
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < (nSpline * piecesExt); l++) {
                coefs[l][j] *= H[l] / (k - j);
            }
        }

        for (int l = 0; l < (nSpline * piecesExt); l++) {
            Q[l % nSpline][l / nSpline] = 0;
            for (int m = 0; m < nSpline; m++) {
                Q[l % nSpline][l / nSpline] += coefs[l][m];
            }
        }

        // cumsum
        for (int l = 0; l < piecesExt; l++) {
            for (int m = 1; m < nSpline; m++) {
                Q[m][l] += Q[m - 1][l];
            }
        }

        for (int l = 0; l < nSpline * piecesExt; l++) {
            if (l % nSpline == 0)
                coefs[l][k] = 0;
            else {
                coefs[l][k] = Q[l % nSpline - 1][l / nSpline];  // questionable
            }
        }

        // normalise antiderivatives by max value
        double fmax[piecesExt * nSpline];
        for (int i = 0; i < piecesExt; i++) {
            for (int j = 0; j < nSpline; j++) {
                fmax[i * nSpline + j] = Q[nSpline - 1][i];
            }
        }

        for (int j = 0; j < k + 1; j++) {
            for (int l = 0; l < nSpline * piecesExt; l++) {
                coefs[l][j] /= fmax[l];
            }
        }

        // diff to adjacent antiderivatives
        for (int i = 0; i < (nSpline * piecesExt) - deg; i++) {
            for (int j = 0; j < k + 1; j++) {
                coefs[i][j] -= coefs[deg + i][j];
            }
        }
        for (int i = 0; i < nSpline * piecesExt; i += nSpline) {
            coefs[i][k] = 0;
        }
    }

    // scale coefficients
    double scale[nSpline * piecesExt];
    for (int i = 0; i < nSpline * piecesExt; i++) {
        scale[i] = 1;
    }
    for (int k = 0; k < nSpline - 1; k++) {
        for (int i = 0; i < (nSpline * piecesExt); i++) {
            scale[i] /= H[i];
        }
        for (int i = 0; i < (nSpline * piecesExt); i++) {
            coefs[i][(nSpline - 1) - (k + 1)] *= scale[i];
        }
    }

    // reduce pieces and sort coefficients by interval number
    int jj[nSpline][pieces];
    for (int i = 0; i < nSpline; i++) {
        for (int j = 0; j < pieces; j++) {
            if (i == 0)
                jj[i][j] = nSpline * (1 + j);
            else
                jj[i][j] = deg;
        }
    }

    for (int i = 1; i < nSpline; i++) {
        for (int j = 0; j < pieces; j++) {
            jj[i][j] += jj[i - 1][j];
        }
    }

    double coefsOut[nSpline * pieces][nSpline];
    int jj_flat;
    for (int i = 0; i < nSpline * pieces; i++) {
        jj_flat = jj[i % nSpline][i / nSpline] - 1;
        for (int j = 0; j < nSpline; j++) {
            coefsOut[i][j] = coefs[jj_flat][j];
        }
    }

    // x-values for B-splines
    int *xsB = malloc((size * nSpline) * sizeof(int));
    int *indexB = malloc((size * nSpline) * sizeof(int));

    int breakInd = 1;
    for (int i = 0; i < size; i++) {
        if (i >= breaks[breakInd] && breakInd < nBreaks - 1) breakInd += 1;
        for (int j = 0; j < nSpline; j++) {
            xsB[i * nSpline + j] = i - breaks[breakInd - 1];
            indexB[i * nSpline + j] = j + (breakInd - 1) * nSpline;
        }
    }

    double *vB = malloc((size * nSpline) * sizeof(double));
    for (int i = 0; i < size * nSpline; i++) {
        vB[i] = coefsOut[indexB[i]][0];
    }

    for (int i = 1; i < nSpline; i++) {
        for (int j = 0; j < size * nSpline; j++) {
            vB[j] = vB[j] * xsB[j] + coefsOut[indexB[j]][i];
        }
    }

    double *A = malloc(size * (nSpline + 1) * sizeof(double));

    for (int i = 0; i < (nSpline + 1) * size; i++) {
        A[i] = 0;
    }
    breakInd = 0;
    for (int i = 0; i < nSpline * size; i++) {
        if (i / nSpline >= breaks[1]) breakInd = 1;
        A[(i % nSpline) + breakInd + (i / nSpline) * (nSpline + 1)] = vB[i];
    }

    double *x = malloc((nSpline + 1) * sizeof(double));
    lsqsolve_sub(size, nSpline + 1, A, size, y, x);

    // coeffs of B-splines to combine by optimised weighting in x
    double C[pieces + nSpline - 1][nSpline * pieces];
    // initialise to 0
    for (int i = 0; i < nSpline + 1; i++) {
        for (int j = 0; j < nSpline * pieces; j++) {
            C[i][j] = 0;
        }
    }

    int CRow, CCol, coefRow, coefCol;
    for (int i = 0; i < nSpline * nSpline * pieces; i++) {
        CRow = i % nSpline + (i / nSpline) % 2;
        CCol = i / nSpline;

        coefRow = i % (nSpline * 2);
        coefCol = i / (nSpline * 2);

        C[CRow][CCol] = coefsOut[coefRow][coefCol];
    }

    // final coefficients
    double coefsSpline[pieces][nSpline];
    for (int i = 0; i < pieces; i++) {
        for (int j = 0; j < nSpline; j++) {
            coefsSpline[i][j] = 0;
        }
    }

    // multiply with x
    for (int j = 0; j < nSpline * pieces; j++) {
        coefCol = j / pieces;
        coefRow = j % pieces;

        for (int i = 0; i < nSpline + 1; i++) {
            coefsSpline[coefRow][coefCol] += C[i][j] * x[i];
        }
    }

    // compute piecewise polynomial
    int secondHalf = 0;
    for (int i = 0; i < size; i++) {
        secondHalf = i < breaks[1] ? 0 : 1;
        yOut[i] = coefsSpline[secondHalf][0];
    }

    for (int i = 1; i < nSpline; i++) {
        for (int j = 0; j < size; j++) {
            secondHalf = j < breaks[1] ? 0 : 1;
            yOut[j] =
                yOut[j] * (j - breaks[1] * secondHalf) + coefsSpline[secondHalf][i];
        }
    }
    free(xsB);
    free(indexB);
    free(vB);
    free(A);
    free(x);

    return 0;
}

double periodicity_wang_th0_01(double *x, int length) {
    const double th = 0.01;

    double *y_spline = malloc(length * sizeof(double));

    // fit a spline with 3 nodes to the data
    splinefit(x, length, y_spline);

    // printf("spline fit complete.\n");

    // subtract spline from data to remove trend
    double *ySub = malloc(length * sizeof(double));
    for (int i = 0; i < length; i++) {
        ySub[i] = x[i] - y_spline[i];
        // printf("ySub[%i] = %1.5f\n", i, ySub[i]);
    }

    // compute autocorrelations up to 1/3 of the length of the time series
    int acmax = (int)ceil((double)length / 3);

    double *acf = malloc(acmax * sizeof(double));
    for (int tau = 1; tau <= acmax; tau++) {
        // correlation/ covariance the same, don't care for scaling (cov would be more
        // efficient)
        acf[tau - 1] = autocov_lag(ySub, length, tau);
        // printf("acf[%i] = %1.9f\n", tau-1, acf[tau-1]);
    }

    // printf("ACF computed.\n");

    // find troughts and peaks
    double *troughs = malloc(acmax * sizeof(double));
    double *peaks = malloc(acmax * sizeof(double));
    int nTroughs = 0;
    int nPeaks = 0;
    double slopeIn = 0;
    double slopeOut = 0;
    for (int i = 1; i < acmax - 1; i++) {
        slopeIn = acf[i] - acf[i - 1];
        slopeOut = acf[i + 1] - acf[i];

        if (slopeIn < 0 && slopeOut > 0) {
            // printf("trough at %i\n", i);
            troughs[nTroughs] = i;
            nTroughs += 1;
        } else if (slopeIn > 0 && slopeOut < 0) {
            // printf("peak at %i\n", i);
            peaks[nPeaks] = i;
            nPeaks += 1;
        }
    }

    // printf("%i troughs and %i peaks found.\n", nTroughs, nPeaks);

    // search through all peaks for one that meets the conditions:
    // (a) a trough before it
    // (b) difference between peak and trough is at least 0.01
    // (c) peak corresponds to positive correlation
    int iPeak = 0;
    double thePeak = 0;
    int iTrough = 0;
    double theTrough = 0;

    int out = 0;

    for (int i = 0; i < nPeaks; i++) {
        iPeak = peaks[i];
        thePeak = acf[iPeak];

        // printf("i=%i/%i, iPeak=%i, thePeak=%1.3f\n", i, nPeaks-1, iPeak, thePeak);

        // find trough before this peak
        int j = -1;
        while (troughs[j + 1] < iPeak && j + 1 < nTroughs) {
            // printf("j=%i/%i, iTrough=%i, theTrough=%1.3f\n", j+1, nTroughs-1,
            // (int)troughs[j+1], acf[(int)troughs[j+1]]);
            j++;
        }
        if (j == -1) continue;

        iTrough = troughs[j];
        theTrough = acf[iTrough];

        // (a) should be implicit

        // (b) different between peak and trough it as least 0.01
        if (thePeak - theTrough < th) continue;

        // (c) peak corresponds to positive correlation
        if (thePeak < 0) continue;

        // use this frequency that first fulfils all conditions.
        out = iPeak;
        break;
    }

    // printf("Before freeing stuff.\n");

    free(y_spline);
    free(ySub);
    free(acf);
    free(troughs);
    free(peaks);

    return out;
}

double embed2_dist_tau_d_expfit_meandiff(double *x, double *ac, int length) {
    int tau = _co_firstzero(ac, length, length);

    double max_tau = (double)length / 10;
    if (tau > max_tau) {
        tau = floor(max_tau);
    }

    double *x_lag = malloc((length - tau) * sizeof(double));
    int x_lag_length = length - tau - 1;
    if (x_lag_length - 1 == 0) {
        return 0;
    }
    for (int i = 0; i < x_lag_length; i++) {
        x_lag[i] = sqrt((x[i + 1] - x[i]) * (x[i + 1] - x[i]) +
                        (x[i + tau] - x[i + tau + 1]) * (x[i + tau] - x[i + tau + 1]));
        if (isnan(x_lag[i])) {
            free(x_lag);
            return NAN;
        }
    }

    double x_lag_mean = _mean(x_lag, x_lag_length);
    double x_lag_std = 0.0;
    double x_lag_max = -INFINITY;
    double x_lag_min = INFINITY;
    for (int i = 0; i < x_lag_length; i++) {
        double v = x_lag[i];
        double diff = v - x_lag_mean;
        x_lag_std += diff * diff;
        if (v > x_lag_max) x_lag_max = v;
        if (v < x_lag_min) x_lag_min = v;
    }

    x_lag_std = sqrt(x_lag_std / (x_lag_length - 1));

    if (x_lag_std < 0.001) {
        return 0;
    }
    int n_bins =
        ceil((x_lag_max - x_lag_min) / (3.5 * x_lag_std / pow(x_lag_length, 1 / 3.)));

    int *bin_count = malloc(n_bins * sizeof(int));
    double *bin_edges = malloc((n_bins + 1) * sizeof(double));
    histcount(x_lag, x_lag_length, n_bins, bin_count, bin_edges);
    double *bin_count_norm = malloc(n_bins * sizeof(double));
    for (int i = 0; i < n_bins; i++) {
        bin_count_norm[i] = (double)bin_count[i] / (double)(x_lag_length);
    }

    double *d_expfit_diff = malloc(n_bins * sizeof(double));
    for (int i = 0; i < n_bins; i++) {
        double expf =
            exp(-(bin_edges[i] + bin_edges[i + 1]) * 0.5 / x_lag_mean) / x_lag_mean;
        if (expf < 0) {
            expf = 0;
        }
        d_expfit_diff[i] = fabs(bin_count_norm[i] - expf);
    }

    double out = _mean(d_expfit_diff, n_bins);
    free(x_lag);
    free(d_expfit_diff);
    free(bin_edges);
    free(bin_count_norm);
    free(bin_count);

    return out;
}

double _corr(double *x, double *y, int length) {
    double nom = 0;
    double denomX = 0;
    double denomY = 0;

    double meanX = _mean(x, length);
    double meanY = _mean(y, length);

    for (int i = 0; i < length; i++) {
        nom += (x[i] - meanX) * (y[i] - meanY);
        denomX += (x[i] - meanX) * (x[i] - meanX);
        denomY += (y[i] - meanY) * (y[i] - meanY);
    }

    return denomX * denomY > 0 ? nom / sqrt(denomX * denomY) : 0;
}

double _autocorr_lag(double *x, int size, int lag) {
    return _corr(x, &(x[lag]), size - lag);
}

double auto_mutual_info_stats_gaussian_fmmi(double *x, int length, int tau) {
    if (tau > ceil((double)length / 2)) {
        tau = ceil((double)length / 2);
    }

    // compute autocorrelations and compute automutual information
    double *ami = malloc(length * sizeof(double));
    for (int i = 0; i < tau; i++) {
        double ac = _autocorr_lag(x, length, i + 1);
        ami[i] = -0.5 * log(1 - ac * ac);
    }

    // find first minimum of automutual information
    double fmmi = tau;
    for (int i = 1; i < tau - 1; i++) {
        if (ami[i] < ami[i - 1] && ami[i] < ami[i + 1]) {
            fmmi = i;
            break;
        }
    }

    free(ami);

    return fmmi;
}

double _median(const double a[], const int size) {
    double m;
    double *b = malloc(size * sizeof *b);
    memcpy(b, a, size * sizeof *b);
    _sort_double(b, size);
    if (size % 2 == 1) {
        m = b[size / 2];
    } else {
        int m1 = size / 2;
        int m2 = m1 - 1;
        m = (b[m1] + b[m2]) / (double)2.0;
    }
    free(b);
    return m;
}

double outlier_include_np_mdrmd(double *x, int length, int sign, double inc) {
    int n_signed = 0;
    double *x_work = malloc(length * sizeof(double));

    // apply sign and check constant time series
    int constantFlag = 1;
    for (int i = 0; i < length; i++) {
        if (x[i] != x[0]) {
            constantFlag = 0;
        }

        // apply sign, save in new variable
        x_work[i] = sign * x[i];

        // count pos/ negs
        if (x_work[i] >= 0) {
            n_signed += 1;
        }
    }
    if (constantFlag) return 0;

    double maxVal = -INFINITY;
    for (int i = 0; i < length; i++)
        if (x_work[i] > maxVal) maxVal = x_work[i];

    // maximum value too small? return 0
    if (maxVal < inc) {
        return 0;
    }

    int threshold = maxVal / inc + 1;

    // save the indices where y > threshold
    double *r = malloc(length * sizeof(double));

    // save the median over indices with absolute value > threshold
    double *msDti1 = malloc(threshold * sizeof(double));
    double *msDti3 = malloc(threshold * sizeof(double));
    double *msDti4 = malloc(threshold * sizeof(double));
    double *Dt_exc = malloc(length * sizeof(double));
    for (int j = 0; j < threshold; j++) {
        int highSize = 0;
        for (int i = 0; i < length; i++) {
            if (x_work[i] >= j * inc) {
                r[highSize] = i + 1;
                highSize += 1;
            }
        }

        for (int i = 0; i < highSize - 1; i++) {
            Dt_exc[i] = r[i + 1] - r[i];
        }

        msDti1[j] = _mean(Dt_exc, highSize - 1);
        msDti3[j] = (highSize - 1) * 100.0 / n_signed;
        msDti4[j] = _median(r, highSize) / ((double)length / 2) - 1;
    }
    free(Dt_exc);

    int trimthr = 2;
    int mj = 0;
    int fbi = threshold - 1;
    for (int i = 0; i < threshold; i++) {
        if (msDti3[i] > trimthr) {
            mj = i;
        }
        if (isnan(msDti1[threshold - 1 - i])) {
            fbi = threshold - 1 - i;
        }
    }

    int trimLimit = mj < fbi ? mj : fbi;
    double outputScalar = _median(msDti4, trimLimit + 1);

    free(r);
    free(x_work);
    free(msDti1);
    free(msDti3);
    free(msDti4);

    return outputScalar;
}

void _cumsum(double *x, int size, double *out) {
    out[0] = x[0];
    for (int i = 1; i < size; i++) {
        out[i] = x[i] + out[i - 1];
    }
}

double summaries_welch_rect(double *x, int length, int what, double *S, double *f,
                            int n_welch) {
    // angualr frequency and spectrum on that
    double *w = malloc(n_welch * sizeof(double));
    double *Sw = malloc(n_welch * sizeof(double));

    double PI = 3.14159265359;
    for (int i = 0; i < n_welch; i++) {
        w[i] = 2 * PI * f[i];
        Sw[i] = S[i] / (2 * PI);
        if (isinf(Sw[i]) | isinf(-Sw[i])) {
            return 0;
        }
    }

    double dw = w[1] - w[0];

    double *csS = malloc(n_welch * sizeof(double));
    _cumsum(Sw, n_welch, csS);

    double output = 0;
    if (what == 0) {
        double csSThres = csS[n_welch - 1] * 0.5;
        double centroid = 0;
        for (int i = 0; i < n_welch; i++) {
            if (csS[i] > csSThres) {
                centroid = w[i];
                break;
            }
        }

        output = centroid;

    } else if (what == 1) {
        double area_5_1 = 0;
        for (int i = 0; i < n_welch / 5; i++) {
            area_5_1 += Sw[i];
        }
        area_5_1 *= dw;

        output = area_5_1;
    }

    free(w);
    free(Sw);
    free(csS);
    return output;
}

double _f_entropy(double *x, int length) {
    double f = 0.0;
    for (int i = 0; i < length; i++) {
        if (x[i] > 0) {
            f += x[i] * log(x[i]);
        }
    }
    return -1 * f;
}

void _subset(int *x, int *y, int start, int end) {
    int j = 0;
    for (int i = start; i < end; i++) {
        y[j++] = x[i];
    }
    return;
}

double motif_three_quantile_hh(double *x, int size) {
    int tmp_idx, r_idx;
    int dynamic_idx;
    int alphabet_size = 3;
    int array_size;
    int *yt = malloc(size * sizeof(int));
    double hh;
    double *out = malloc(124 * sizeof(double));

    _sb_coarsegrain(x, size, 3, yt);

    // words of length 1
    array_size = alphabet_size;
    int **r1 = malloc(array_size * sizeof(*r1));
    int *sizes_r1 = malloc(array_size * sizeof(sizes_r1));
    double *out1 = malloc(array_size * sizeof(out1));
    for (int i = 0; i < alphabet_size; i++) {
        r1[i] = malloc(size * sizeof(r1[i]));
        r_idx = 0;
        sizes_r1[i] = 0;
        for (int j = 0; j < size; j++) {
            if (yt[j] == i + 1) {
                r1[i][r_idx++] = j;
                sizes_r1[i]++;
            }
        }
    }

    // words of length 2
    array_size *= alphabet_size;

    for (int i = 0; i < alphabet_size; i++) {
        if (sizes_r1[i] != 0 && r1[i][sizes_r1[i] - 1] == size - 1) {
            // int * tmp_ar = malloc((sizes_r1[i] - 1) * sizeof(tmp_ar));
            int *tmp_ar = malloc(sizes_r1[i] * sizeof(tmp_ar));
            _subset(r1[i], tmp_ar, 0, sizes_r1[i]);
            memcpy(r1[i], tmp_ar, (sizes_r1[i] - 1) * sizeof(tmp_ar));
            sizes_r1[i]--;
            free(tmp_ar);
        }
    }

    int ***r2 = malloc(alphabet_size * sizeof(**r2));
    int **sizes_r2 = malloc(alphabet_size * sizeof(*sizes_r2));
    double **out2 = malloc(alphabet_size * sizeof(*out2));

    // allocate separately
    for (int i = 0; i < alphabet_size; i++) {
        r2[i] = malloc(alphabet_size * sizeof(*r2[i]));
        sizes_r2[i] = malloc(alphabet_size * sizeof(*sizes_r2[i]));
        // out2[i] = malloc(alphabet_size * sizeof(out2[i]));
        out2[i] = malloc(alphabet_size * sizeof(**out2));
        for (int j = 0; j < alphabet_size; j++) {
            r2[i][j] = malloc(size * sizeof(*r2[i][j]));
        }
    }

    // fill separately
    for (int i = 0; i < alphabet_size; i++) {
        // for (int i = 0; i < array_size; i++) {
        // r2[i] = malloc(alphabet_size * sizeof(r2[i]));
        // sizes_r2[i] = malloc(alphabet_size * sizeof(sizes_r2[i]));
        // out2[i] = malloc(alphabet_size * sizeof(out2[i]));
        for (int j = 0; j < alphabet_size; j++) {
            // r2[i][j] = malloc(size * sizeof(r2[i][j]));
            sizes_r2[i][j] = 0;
            dynamic_idx = 0;  // workaround as you can't just add elements to array
            // like in python (list.append()) for example, so since for some k there
            // will be no adding, you need to keep track of the idx at which elements
            // will be inserted
            for (int k = 0; k < sizes_r1[i]; k++) {
                tmp_idx = yt[r1[i][k] + 1];
                if (tmp_idx == (j + 1)) {
                    r2[i][j][dynamic_idx++] = r1[i][k];
                    sizes_r2[i][j]++;
                    // printf("dynamic_idx=%i, size = %i\n", dynamic_idx, size);
                }
            }
            double tmp = (double)sizes_r2[i][j] / ((double)(size) - (double)(1.0));
            out2[i][j] = tmp;
        }
    }

    hh = 0.0;
    for (int i = 0; i < alphabet_size; i++) {
        hh += _f_entropy(out2[i], alphabet_size);
    }

    free(yt);
    free(out);

    free(sizes_r1);

    // free nested array
    for (int i = 0; i < alphabet_size; i++) {
        free(r1[i]);
    }
    free(r1);
    // free(sizes_r1);

    for (int i = 0; i < alphabet_size; i++) {
        // for (int i = alphabet_size - 1; i >= 0; i--) {

        free(sizes_r2[i]);
        free(out2[i]);
    }

    // for (int i = alphabet_size-1; i >= 0 ; i--) {
    for (int i = 0; i < alphabet_size; i++) {
        for (int j = 0; j < alphabet_size; j++) {
            free(r2[i][j]);
        }
        free(r2[i]);
    }

    free(r2);
    free(sizes_r2);
    free(out2);

    return hh;
}

int _linreg(int n, double *x, double *y, double *m, double *b) {
    double sumx = 0.0;  /* sum of x     */
    double sumx2 = 0.0; /* sum of x**2  */
    double sumxy = 0.0; /* sum of x * y */
    double sumy = 0.0;  /* sum of y     */
    double sumy2 = 0.0; /* sum of y**2  */

    for (int i = 0; i < n; i++) {
        sumx += x[i];
        sumx2 += x[i] * x[i];
        sumxy += x[i] * y[i];
        sumy += y[i];
        sumy2 += y[i] * y[i];
    }

    double denom = (n * sumx2 - sumx * sumx);
    if (denom == 0) {
        *m = 0;
        *b = 0;
        return 1;
    }

    *m = (n * sumxy - sumx * sumy) / denom;
    *b = (sumy * sumx2 - sumx * sumxy) / denom;

    return 0;
}

double _norm(double *x, int length) {
    double out = 0.0;
    for (int i = 0; i < length; i++) {
        out += x[i] * x[i];
    }
    return sqrt(out);
}

double fluct_anal_2_50_1_logi_prop_r1(double *y, int size, int lag, int how) {
    // generate log spaced tau vector
    double linLow = log(5);
    double linHigh = log(size / 2);

    int nTauSteps = 50;
    double tauStep = (linHigh - linLow) / (nTauSteps - 1);

    int tau[50];
    for (int i = 0; i < nTauSteps; i++) {
        tau[i] = round(exp(linLow + i * tauStep));
    }

    // check for uniqueness, use ascending order
    int nTau = nTauSteps;
    for (int i = 0; i < nTauSteps - 1; i++) {
        while (tau[i] == tau[i + 1] && i < nTau - 1) {
            for (int j = i + 1; j < nTauSteps - 1; j++) {
                tau[j] = tau[j + 1];
            }
            // lost one
            nTau -= 1;
        }
    }

    // fewer than 12 points -> leave.
    if (nTau < 12) {
        return 0;
    }

    int sizeCS = size / lag;
    double *yCS = malloc(sizeCS * sizeof(double));

    // transform input vector to cumsum
    yCS[0] = y[0];
    for (int i = 0; i < sizeCS - 1; i++) {
        yCS[i + 1] = yCS[i] + y[(i + 1) * lag];
    }

    // for each value of tau, cut signal into snippets of length tau, detrend and
    // first generate a support for regression (detrending)
    double *xReg = malloc(tau[nTau - 1] * sizeof *xReg);
    for (int i = 0; i < tau[nTau - 1]; i++) {
        xReg[i] = i + 1;
    }

    // iterate over taus, cut signal, detrend and save amplitude of remaining signal
    double *F = malloc(nTau * sizeof *F);
    for (int i = 0; i < nTau; i++) {
        int nBuffer = sizeCS / tau[i];
        double *buffer = malloc(tau[i] * sizeof *buffer);
        double m = 0.0, b = 0.0;

        F[i] = 0;
        for (int j = 0; j < nBuffer; j++) {
            _linreg(tau[i], xReg, yCS + j * tau[i], &m, &b);
            for (int k = 0; k < tau[i]; k++) {
                buffer[k] = yCS[j * tau[i] + k] - (m * (k + 1) + b);
            }

            if (how == 0) {
                double max_val = -INFINITY;
                double min_val = INFINITY;
                for (int k = 0; k < tau[i]; k++) {
                    if (buffer[k] > max_val) max_val = buffer[k];
                    if (buffer[k] < min_val) min_val = buffer[k];
                }
                double diff = max_val - min_val;
                F[i] += diff * diff;
            } else if (how == 1) {
                for (int k = 0; k < tau[i]; k++) {
                    F[i] += buffer[k] * buffer[k];
                }
            } else {
                return 0.0;
            }
        }

        if (how == 0) {
            F[i] = sqrt(F[i] / nBuffer);
        } else if (how == 1) {
            F[i] = sqrt(F[i] / (nBuffer * tau[i]));
        }

        free(buffer);
    }

    double *logtt = malloc(nTau * sizeof *logtt);
    double *logFF = malloc(nTau * sizeof *logFF);
    int ntt = nTau;

    for (int i = 0; i < nTau; i++) {
        logtt[i] = log(tau[i]);
        logFF[i] = log(F[i]);
    }

    int minPoints = 6;
    int nsserr = (ntt - 2 * minPoints + 1);
    double *sserr = malloc(nsserr * sizeof *sserr);
    double *buffer = malloc((ntt - minPoints + 1) * sizeof *buffer);
    for (int i = minPoints; i < ntt - minPoints + 1; i++) {
        double m1 = 0.0, b1 = 0.0;
        double m2 = 0.0, b2 = 0.0;

        sserr[i - minPoints] = 0.0;

        _linreg(i, logtt, logFF, &m1, &b1);
        _linreg(ntt - i + 1, logtt + i - 1, logFF + i - 1, &m2, &b2);

        for (int j = 0; j < i; j++) {
            buffer[j] = logtt[j] * m1 + b1 - logFF[j];
        }

        sserr[i - minPoints] += _norm(buffer, i);

        for (int j = 0; j < ntt - i + 1; j++) {
            buffer[j] = logtt[j + i - 1] * m2 + b2 - logFF[j + i - 1];
        }

        sserr[i - minPoints] += _norm(buffer, ntt - i + 1);
    }

    double firstMinInd = 0.0;
    double minimum = INFINITY;
    for (int i = 0; i < nsserr; i++) {
        if (sserr[i] < minimum) minimum = sserr[i];
    }

    for (int i = 0; i < nsserr; i++) {
        if (sserr[i] == minimum) {
            firstMinInd = i + minPoints - 1;
            break;
        }
    }

    free(yCS);  // new

    free(xReg);
    free(F);
    free(logtt);
    free(logFF);
    free(sserr);
    free(buffer);

    return (firstMinInd + 1) / ntt;
}