# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from libc.math cimport INFINITY, floor, log2, pow
from libc.stdlib cimport free, malloc
from numpy cimport uint32_t

from ..utils cimport TSArray
from ..utils._misc cimport to_ndarray_int
from ..utils._rand cimport rand_int, rand_normal, rand_uniform
from ._feature cimport Feature, FeatureEngineer


cdef struct Rocket:
    Py_ssize_t length
    Py_ssize_t dilation
    Py_ssize_t padding
    bint return_mean
    double bias
    double *weight

cdef class KernelSampler:

    cdef void sample(
        self,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double *weights,
        Py_ssize_t length,
        double *mean,
        uint32_t *seed
    ) noexcept nogil:
        pass


cdef class NormalKernelSampler(KernelSampler):
    cdef double mean
    cdef double scale

    def __init__(self, mean=0.0, scale=1.0):
        self.mean = mean
        self.scale = scale

    cdef void sample(
        self,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double *weights,
        Py_ssize_t length,
        double *mean,
        uint32_t *seed
    ) noexcept nogil:
        cdef Py_ssize_t i
        mean[0] = 0
        for i in range(length):
            weights[i] = rand_normal(self.mean, self.scale, seed)
            mean[0] += weights[i]

        mean[0] = mean[0] / length


cdef class UniformKernelSampler(KernelSampler):

    cdef double lower
    cdef double upper

    def __init__(self, lower=-1.0, upper=1.0):
        self.lower = lower
        self.upper = upper

    cdef void sample(
        self,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double *weights,
        Py_ssize_t length,
        double *mean,
        uint32_t *seed
    ) noexcept nogil:
        cdef Py_ssize_t i
        mean[0] = 0
        for i in range(length):
            weights[i] = rand_uniform(self.lower, self.upper, seed)
            mean[0] += weights[i]

        mean[0] = mean[0] / length


cdef class ShapeletKernelSampler(KernelSampler):
    cdef void sample(
        self,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double *weights,
        Py_ssize_t length,
        double *mean,
        uint32_t *seed
    ) noexcept nogil:
        cdef Py_ssize_t start
        cdef Py_ssize_t index
        cdef Py_ssize_t dim
        cdef Py_ssize_t i

        start = rand_int(0, X.shape[2] - length, seed)
        index = samples[rand_int(0, n_samples, seed)]
        if X.shape[1] > 1:
            dim = rand_int(0, X.shape[1], seed)
        else:
            dim = 1

        mean[0] = 0
        cdef const double *data = &X[index, dim, start]
        for i in range(length):
            weights[i] = data[i]
            mean[0] += weights[i]

        mean[0] /= length


cdef class RocketFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t n_kernels
    cdef KernelSampler weight_sampler
    cdef double padding_prob
    cdef double bias_prob
    cdef double normalize_prob
    cdef Py_ssize_t *kernel_size
    cdef Py_ssize_t n_kernel_size

    def __cinit__(
        self,
        Py_ssize_t n_kernels,
        KernelSampler weight_sampler,
        object kernel_size,
        double bias_prob,
        double padding_prob,
        double normalize_prob
    ):
        self.n_kernels = n_kernels
        self.weight_sampler = weight_sampler
        self.kernel_size = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * len(kernel_size))
        self.n_kernel_size = len(kernel_size)

        cdef Py_ssize_t i
        for i in range(len(kernel_size)):
            self.kernel_size[i] = kernel_size[i]

        self.bias_prob = bias_prob
        self.padding_prob = padding_prob
        self.normalize_prob = normalize_prob

    def __reduce__(self):
        return self.__class__, (
            self.n_kernels,
            self.weight_sampler,
            to_ndarray_int(self.kernel_size, self.n_kernel_size),
            self.bias_prob,
            self.padding_prob,
            self.normalize_prob
        )

    def __dealloc__(self):
        free(self.kernel_size)

    cdef Py_ssize_t get_n_features(self, TSArray X) noexcept nogil:
        return self.n_kernels

    cdef Py_ssize_t get_n_outputs(self, TSArray X) noexcept nogil:
        return self.get_n_features(X) * 2

    cdef Py_ssize_t next_feature(
        self,
        Py_ssize_t feature_id,
        TSArray X, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        Feature *transient,
        uint32_t *seed
    ) noexcept nogil:
        cdef Rocket *rocket = <Rocket*> malloc(sizeof(Rocket))
        cdef Py_ssize_t i
        cdef double mean
        cdef Py_ssize_t length = self.kernel_size[rand_int(0, self.n_kernel_size, seed)]
        cdef double* weight = <double*> malloc(sizeof(double) * length)

        self.weight_sampler.sample(
            X, samples, n_samples, weight, length, &mean, seed
        )
        if rand_uniform(0, 1, seed) < self.normalize_prob:
            for i in range(length):
                weight[i] -= mean

        rocket.length = length
        rocket.dilation = <Py_ssize_t> floor(
            pow(2, rand_uniform(0, log2((X.shape[2] - 1) / (rocket.length - 1)), seed))
        )
        rocket.padding = 0
        if rand_uniform(0, 1, seed) < self.padding_prob:
            rocket.padding = ((rocket.length - 1) * rocket.dilation) // 2

        rocket.return_mean = rand_uniform(0, 1, seed) < 0.5
        rocket.weight = weight

        rocket.bias = 0
        if rand_uniform(0, 1, seed) < self.bias_prob:
            rocket.bias = rand_uniform(-1, 1, seed)

        if X.shape[1] > 1:
            transient.dim = rand_int(0, X.shape[1], seed)
        else:
            transient.dim = 0

        transient.feature = rocket
        return 0

    cdef Py_ssize_t free_transient_feature(self, Feature *feature) noexcept nogil:
        return self.free_persistent_feature(feature)

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) noexcept nogil:
        cdef Rocket *rocket
        if feature.feature != NULL:
            rocket = <Rocket*> feature.feature
            if rocket.weight != NULL:
                free(rocket.weight)
            free(feature.feature)
            feature.feature = NULL
        return 0

    # NOTE: We move ownership of `transient.feature` to `persistent.feature`.
    cdef Py_ssize_t init_persistent_feature(
        self, 
        TSArray X,
        Feature *transient, 
        Feature *persistent
    ) noexcept nogil:
        persistent.dim = transient.dim
        persistent.feature = transient.feature
        transient.feature = NULL
        return 0

    cdef double transient_feature_value(
        self,
        Feature *feature,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        cdef double mean_val, max_val
        cdef Rocket* rocket = <Rocket*> feature.feature
        
        # TODO: (1.3) use utils._cconv.convolution_1d
        apply_convolution(
            rocket.dilation,
            rocket.padding,
            rocket.bias,
            rocket.weight,
            rocket.length,
            &X[sample, feature.dim, 0],
            X.shape[2],
            &mean_val,
            &max_val,
        )
        if rocket.return_mean:
            return mean_val
        else:
            return max_val

    cdef double persistent_feature_value(
        self,
        Feature *feature,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        return self.transient_feature_value(feature, X, sample)

    cdef Py_ssize_t transient_feature_fill(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t feature_id,
    ) noexcept nogil:
        cdef double mean_val, max_val
        cdef Rocket* rocket = <Rocket*> feature.feature
        apply_convolution(
            rocket.dilation,
            rocket.padding,
            rocket.bias,
            rocket.weight,
            rocket.length,
            &X[sample, feature.dim, 0],
            X.shape[2],
            &mean_val,
            &max_val,
        )
        cdef Py_ssize_t feature_offset = feature_id * 2
        out[out_sample, feature_offset] = mean_val
        out[out_sample, feature_offset + 1] = max_val
        return 0

    cdef Py_ssize_t persistent_feature_fill(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) noexcept nogil:
        return self.transient_feature_fill(
            feature, X, sample, out, out_sample, out_feature
        )

    cdef object persistent_feature_to_object(self, Feature *feature):
        cdef Py_ssize_t j
        cdef Rocket *rocket = <Rocket*> feature.feature

        weights = np.empty(rocket.length, dtype=float)
        for j in range(rocket.length):
            weights[j] = rocket.weight[j]

        return feature.dim, (
            rocket.length,
            rocket.dilation,
            rocket.padding,
            weights,
            rocket.bias,
            rocket.return_mean
        )

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature):
        dim, (length, dilation, padding, weight, bias, return_mean) = object

        cdef Rocket *rocket = <Rocket*> malloc(sizeof(Rocket))
        rocket.length = length
        rocket.dilation = dilation
        rocket.padding = padding
        rocket.bias = bias
        rocket.weight = <double*> malloc(sizeof(double) * length)
        rocket.return_mean = return_mean
        
        cdef Py_ssize_t i
        for i in range(length):
            rocket.weight[i] = weight[i]

        feature.feature = rocket
        feature.dim = dim
        return 0

# TODO(1.3): Remove
cdef void apply_convolution(
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double bias,
    double *weight,
    Py_ssize_t length,
    const double* x,
    Py_ssize_t x_length,
    double* mean_val,
    double* max_val
) noexcept nogil:
    cdef Py_ssize_t out_len, end
    cdef Py_ssize_t i, j, k
    cdef double ppv
    cdef double max
    cdef double sum

    out_len = (x_length + 2 * padding) - ((length - 1) * dilation)
    end = (x_length + padding) - ((length - 1) * dilation)
    max_val[0] = -INFINITY
    mean_val[0] = 0.0
    for i in range(-padding, end):
        inner_prod = bias
        k = i
        for j in range(length):
            if -1 < k < x_length:
                inner_prod += weight[j] * x[k]
            k += dilation
        if inner_prod > max_val[0]:
            max_val[0] = inner_prod
        if inner_prod > 0:
            mean_val[0] += 1

    mean_val[0] /= out_len
