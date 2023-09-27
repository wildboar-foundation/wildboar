# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

import numpy as np

from libc.math cimport INFINITY, floor, log2, pow, sqrt, fabs
from libc.stdlib cimport free, malloc
from numpy cimport uint32_t

from ._feature cimport Feature, FeatureEngineer

from ..utils cimport TSArray
from ..utils._rand cimport rand_normal

# Hydra group
cdef struct Hydra:
    Py_ssize_t kernel_size
    Py_ssize_t n_kernels

    # size: kernel_size * n_kernels
    double *kernels

cdef class WeightSampler:

    cdef void sample(
        self,
        double *data,
        Py_ssize_t length,
        double *mean,
        double *sum_abs,
        uint32_t *seed
    ) noexcept nogil:
        pass


cdef class NormalWeightSampler(WeightSampler):
    cdef double mean
    cdef double scale

    def __init__(self, mean=0.0, scale=1.0):
        self.mean = mean
        self.scale = scale

    cdef void sample(
        self,
        double *data,
        Py_ssize_t length,
        double *mean,
        double *sum_abs,
        uint32_t *seed
    ) noexcept nogil:
        cdef Py_ssize_t i
        mean[0] = 0
        sum_abs[0] = 0
        for i in range(length):
            data[i] = rand_normal(self.mean, self.scale, seed)
            mean[0] += data[i]
            sum_abs[0] += fabs(data[i])

        mean[0] = mean[0] / length
  
cdef class HydraFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t n_kernels
    cdef Py_ssize_t kernel_size
    cdef Py_ssize_t n_groups
    cdef WeightSampler weight_sampler

    # Temporary buffers for storing values while computing the convolution.
    cdef double *conv_values
    
    def __cinit__(
        self,
        Py_ssize_t n_groups,
        Py_ssize_t n_kernels,
        Py_ssize_t kernel_size,
        WeightSampler weight_sampler,
    ):
        self.n_groups = n_groups
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.weight_sampler = weight_sampler
        self.conv_values = NULL

    def __dealloc__(self):
        self._free() 

    cdef void _free(self) noexcept nogil:
        if self.conv_values != NULL:
            free(self.conv_values)

    def __reduce__(self):
        return self.__class__, (
            self.n_groups, self.n_kernels, self.kernel_size, self.weight_sampler
        )

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        self.conv_values = <double*> malloc(sizeof(double) * X.shape[2] - 1)
   
    cdef Py_ssize_t get_n_features(self, TSArray X) noexcept nogil:
        return self.n_groups * self.n_kernels

    # Each timeseries is represented by:
    #   n_groups * n_kernels * 2 (soft_max, hard_min) * 2 (X, diff(X))
    # features.
    cdef Py_ssize_t get_n_outputs(self, TSArray X) noexcept nogil:
        return self.get_n_features(X) * 2 * 1 # soft_max and hard_min and (TODO) X and diff(X)

    cdef Py_ssize_t next_feature(
        self,
        Py_ssize_t feature_id,
        TSArray X, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        Feature *transient,
        uint32_t *seed
    ) noexcept nogil:
        cdef Hydra *hydra = <Hydra*> malloc(sizeof(Hydra))
        cdef Py_ssize_t i
        cdef double mean
        cdef double sum_abs
        cdef double *kernels = <double*> malloc(sizeof(double) * self.kernel_size * self.n_kernels)
        cdef double *kernel
        
        # Randomly sample and rescale n_kernels Hydra (2023), D (fit, Line 4)
        for i in range(self.n_kernels):
            kernel = kernels + i * self.kernel_size
            self.weight_sampler.sample(kernel, self.kernel_size, &mean, &sum_abs, seed) 

            # Hydra (2023), D (fit, Line 6)
            for j in range(self.kernel_size):
                kernel[i] = (kernel[i] - mean) / sum_abs
       
        hydra.kernel_size = self.kernel_size
        hydra.n_kernels = self.n_kernels
        hydra.kernels = kernels

        transient.dim = 0
        transient.feature = hydra
        return 0 
   
    # Restore a Hydra group from a numpy array
    cdef Py_ssize_t persistent_feature_from_object(self, object obj, Feature *feature):
        dim, array = obj
        cdef Hydra *hydra = <Hydra*> malloc(sizeof(Hydra))

        # We will assume that array.shape[0] == hydra.n_kernels * hydra.kernel_size
        hydra.kernels = <double*> malloc(sizeof(double) * array.shape[0])
        cdef Py_ssize_t i
        
        # TODO: This can potentially be done without GIL.
        for i in range(array.shape[0]):
            hydra.kernels[i] = array[i]

        feature.feature = hydra
        feature.dim = dim
        return 0

    # NOTE: feature.feature will be NULL if we have called
    # `init_persistent_feature` which has moved ownership to the persistent
    # feature. Still, we should free the data occupied by the transient feature
    # if ownership has not been transfered.
    cdef Py_ssize_t free_transient_feature(self, Feature *feature) noexcept nogil:
        cdef Hydra *hydra
        if feature.feature != NULL:
            hydra = <Hydra*> feature.feature
            if hydra.kernels != NULL:
                free(hydra.kernels)
            free(feature.feature)
        return 0

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) noexcept nogil:
        return self.free_transient_feature(feature)

    # NOTE: We move ownership of the feature here to a persistent feature, which will
    # be freed by `free_persistent_feature`.
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

    cdef object persistent_feature_to_object(self, Feature *feature):
        cdef Py_ssize_t i
        cdef Hydra *hydra = <Hydra*> feature.feature
        cdef n = hydra.n_kernels * hydra.kernel_size
        data = np.empty(n, dtype=float)
        for i in range(n):
            data[i] = hydra.kernels[i]

        return feature.dim, data

    cdef double transient_feature_value(
        self, Feature *feature, TSArray X, Py_ssize_t sample
    ) noexcept nogil:
       return 0 # TODO

    cdef double persistent_feature_value(
        self, Feature *feature, TSArray X, Py_ssize_t sample
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
        cdef Py_ssize_t dilation, q, i, j
        cdef Py_ssize_t padding
        cdef Py_ssize_t max_exponent = <Py_ssize_t> floor(
            log2((X.shape[2] - 1) / (self.kernel_size - 1))
        )
        cdef Hydra *hydra = <Hydra*> feature.feature
        cdef double *kernel 
        cdef double mean_val
        if max_exponent < 1:
            max_exponent = 1

        # Hydra (2023), D (transform, Line 5)
        for dilation in range(max_exponent + 1):
            padding = ((self.kernel_size - 1) * pow(2, dilation)) // 2

            for i in range(self.n_kernels):
                kernel = hydra.kernels + (i * self.kernel_size)
                apply_convolution(
                    dilation,
                    padding,
                    kernel,
                    self.kernel_size,
                    &X[sample, feature.dim],
                    X.shape[2],
                    self.conv_values,
                )
            
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

cdef void apply_convolution(
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double *weight,
    Py_ssize_t length,
    const double* x,
    Py_ssize_t x_length,
    double* value,
) noexcept nogil:
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t out_len = (x_length + 2 * padding) - ((length - 1) * dilation)
    cdef Py_ssize_t end = (x_length + padding) - ((length - 1) * dilation)
    for i in range(-padding, end):
        inner_prod = 0.0
        k = i
        for j in range(length):
            if -1 < k < x_length:
                inner_prod += weight[j] * x[k]
            k += dilation
        value[i + padding] = inner_prod
