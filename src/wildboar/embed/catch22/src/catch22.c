#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double _covariance(double *x, double *y, size_t length) {
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

double _mean(double *x, size_t length) {
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

double _quantile(double *x_sorted, size_t length, double quant) {
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

void histcount(double *x, size_t size, size_t n_bins, size_t *bin_count,
               double *bin_edges) {
    double min_val = INFINITY;
    double max_val = -INFINITY;
    memset(bin_count, 0, sizeof(size_t) * n_bins);

    for (size_t i = 0; i < size; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
        }
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    double bin_step = (max_val - min_val) / n_bins;
    for (size_t i = 0; i < size; i++) {
        size_t bin = (size_t)((x[i] - min_val) / bin_step);
        if (bin < 0) {
            bin = 0;
        }
        if (bin >= n_bins) {
            bin = n_bins - 1;
        }
        bin_count[bin] += 1;
    }

    for (size_t i = 0; i < n_bins + 1; i++) {
        bin_edges[i] = i * bin_step + min_val;
    }
}

double histogram_mode(double *x, size_t length, size_t *hist_counts, double *bin_edges,
                      size_t n_bins) {
    histcount(x, length, n_bins, hist_counts, bin_edges);

    size_t max_count = 0;
    size_t n_max_value = 1;
    double out = 0;
    for (size_t i = 0; i < n_bins; i++) {
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

void histbinassign(double *x, const size_t length, double *bin_edges, size_t n_edges,
                   size_t *bin) {
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
void histcount_edges(double *x, size_t length, double *bin_edges, size_t n_edges,
                     size_t *histcounts) {
    memset(histcounts, 0, sizeof(size_t) * n_edges);
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < n_edges; j++) {
            if (x[i] <= bin_edges[j]) {
                histcounts[j] += 1;
                break;
            }
        }
    }
}

double histogram_ami_even(double *x, size_t length, size_t tau, size_t n_bins) {
    double *x_lag = malloc((length - tau) * sizeof(double));
    double *x_pad = malloc((length - tau) * sizeof(double));

    for (int i = 0; i < length - tau; i++) {
        x_lag[i] = x[i];
        x_pad[i] = x[i + tau];
    }

    double min_val = INFINITY;
    double max_val = -INFINITY;

    for (size_t i = 0; i < length; i++) {
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

    size_t *lag_bins = (size_t *)malloc(sizeof(size_t) * length - tau);
    size_t *pad_bins = (size_t *)malloc(sizeof(size_t) * length - tau);
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

    size_t *joint_hist_count =
        (size_t *)malloc(sizeof(size_t) * (n_bins + 1) * (n_bins + 1));
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

int _is_constant(double *x, size_t length) {
    for (size_t i = 0; i < length; i++) {
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

void _sb_coarsegrain(double *x, size_t length, const size_t num_groups,
                     size_t *labels) {
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

double transition_matrix_ac_sumdiagcov(double *x, double *ac, size_t length,
                                       size_t n_groups) {
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

    size_t *labels = malloc(n_down * sizeof(size_t));
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

double periodicity_wang_th0_01(double *x, size_t length) {
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

        if (slopeIn<0 & slopeOut> 0) {
            // printf("trough at %i\n", i);
            troughs[nTroughs] = i;
            nTroughs += 1;
        } else if (slopeIn > 0 & slopeOut < 0) {
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

double embed2_dist_tau_d_expfit_meandiff(double *x, double *ac, size_t length) {
    int tau = _co_firstzero(ac, length, length);

    double max_tau = (double)length / 10;
    if (tau > max_tau) {
        tau = floor(max_tau);
    }

    double *x_lag = malloc((length - tau) * sizeof(double));
    size_t x_lag_length = length - tau - 1;
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
    size_t n_bins =
        ceil((x_lag_max - x_lag_min) / (3.5 * x_lag_std / pow(x_lag_length, 1 / 3.)));

    size_t *bin_count = malloc(n_bins * sizeof(size_t));
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

double _corr(double *x, double *y, size_t length) {
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

double _autocorr_lag(double *x, size_t size, size_t lag) {
    return _corr(x, &(x[lag]), size - lag);
}

double auto_mutual_info_stats_gaussian_fmmi(double *x, size_t length, size_t tau) {
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
        if (ami[i] < ami[i - 1] & ami[i] < ami[i + 1]) {
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

double outlier_include_np_mdrmd(double *x, size_t length, int sign, double inc) {
    size_t n_signed = 0;
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
    double *Dt_exc = malloc(threshold * sizeof(double));
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

void _cumsum(const double a[], const int size, double b[]) {
    b[0] = a[0];
    for (int i = 1; i < size; i++) {
        b[i] = a[i] + b[i - 1];
    }
}

double summaries_welch_rect(double *x, size_t length, int what, double *S, double *f,
                            size_t n_welch) {
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