# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

import numpy as np

from libc.math cimport INFINITY, floor, log2, pow, sqrt, fabs
from libc.stdlib cimport free, malloc, labs
from libc.string cimport memset

from numpy cimport uint32_t

from ._feature cimport Feature, FeatureEngineer

from ..utils cimport TSArray
from ..utils._rand cimport rand_normal
from ..utils._cconv cimport convolution_1d

# Hydra group
cdef struct Hydra:
    Py_ssize_t kernel_size
    Py_ssize_t n_kernels

    # size: kernel_size * n_kernels
    double *kernels

cdef class KernelSampler:

    cdef void sample(
        self,
        double *data,
        Py_ssize_t length,
        double *mean,
        double *sum_abs,
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


cdef inline Py_ssize_t _max_exponent(
    Py_ssize_t n_timestep, Py_ssize_t kernel_size
) noexcept nogil:
    cdef Py_ssize_t max_exponent = <Py_ssize_t> floor(
        log2((n_timestep - 1) / <double> (kernel_size - 1))
    )
    if max_exponent < 0:
        max_exponent = 0
    max_exponent += 1

    return max_exponent

  
cdef class HydraFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t n_kernels
    cdef Py_ssize_t kernel_size
    cdef Py_ssize_t n_groups
    cdef KernelSampler kernel_sampler

    # Temporary buffers to store values while computing the convolution.
    cdef double *conv_values

    # Temporary buffers to store the min/max kernel values
    cdef double *min_values
    cdef double *max_values

    def __cinit__(
        self,
        Py_ssize_t n_groups,
        Py_ssize_t n_kernels,
        Py_ssize_t kernel_size,
        KernelSampler kernel_sampler,
    ):
        self.n_groups = n_groups
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.kernel_sampler = kernel_sampler
        self.conv_values = NULL
        self.min_values = NULL
        self.max_values = NULL

    def __dealloc__(self):
        self._free() 

    cdef void _free(self) noexcept nogil:
        if self.conv_values != NULL:
            free(self.conv_values)
            self.conv_values = NULL
        if self.min_values != NULL:
            free(self.min_values)
            self.min_values = NULL
        if self.max_values != NULL:
            free(self.max_values)
            self.max_values = NULL

    def __reduce__(self):
        return self.__class__, (
            self.n_groups, self.n_kernels, self.kernel_size, self.kernel_sampler
        )

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        self.conv_values = <double*> malloc(sizeof(double) * X.shape[2] * self.n_kernels)
        self.max_values = <double*> malloc(sizeof(double) * self.n_kernels)
        self.min_values = <double*> malloc(sizeof(double) * self.n_kernels)
   
    cdef Py_ssize_t get_n_features(self, TSArray X) noexcept nogil:
        return self.n_groups

    # Each timeseries is represented by:
    #   n_groups * n_kernels * max_exponent * 2 (soft_max, hard_min) * 2 (X, diff(X))
    # features.
    #
    # TODO: implement support for diff(X)
    cdef Py_ssize_t get_n_outputs(self, TSArray X) noexcept nogil:
        cdef Py_ssize_t max_exponent = _max_exponent(X.shape[2], self.kernel_size)
        return self.get_n_features(X) * max_exponent * self.n_kernels * 2 * 1 # soft_max and hard_min and (TODO) X and diff(X)

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
        cdef Py_ssize_t i, j
        cdef double mean
        cdef double sum_abs
        cdef double *kernels = <double*> malloc(sizeof(double) * self.kernel_size * self.n_kernels)
        cdef double *kernel
        
        # Randomly sample and rescale n_kernels Hydra (2023), D (fit, Line 4)
        for i in range(self.n_kernels):
            kernel = kernels + i * self.kernel_size
            self.kernel_sampler.sample(kernel, self.kernel_size, &mean, &sum_abs, seed) 

            # Hydra (2023), D (fit, Line 6)
            for j in range(self.kernel_size):
                kernel[j] = (kernel[j] - mean) / sum_abs
       
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
    # feature.
    cdef Py_ssize_t free_transient_feature(self, Feature *feature) noexcept nogil:
        return 0

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) noexcept nogil:
        cdef Hydra *hydra
        if feature.feature != NULL:
            hydra = <Hydra*> feature.feature
            if hydra.kernels != NULL:
                free(hydra.kernels)
                hydra.kernels = NULL
            free(feature.feature)
            feature.feature = NULL
        return 0 

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
        # transient.feature = NULL
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
        cdef Py_ssize_t max_exponent = _max_exponent(X.shape[2], self.kernel_size)
        cdef Py_ssize_t i, exponent, padding, dilation
        cdef Py_ssize_t kernel_feature_offset

        cdef Py_ssize_t min_index, max_index
        cdef double min_value, max_value

        cdef Hydra *hydra = <Hydra*> feature.feature

        # Place the pointer inside correct feature group, as given by feature_id
        cdef Py_ssize_t feature_offset = (
            feature_id * self.n_kernels * max_exponent * 2
        )

        # Hydra (2023), D (transform, Line 5)
        for exponent in range(max_exponent):
            dilation = <Py_ssize_t> pow(2, exponent)
            padding = <Py_ssize_t> ((self.kernel_size - 1) * dilation) // 2

            # We store the output of the convolution in an array
            # of shape X.shape[2] * self.n_kernels
            for i in range(self.n_kernels):
                convolution_1d(
                    1,
                    dilation,
                    padding,
                    0.0,
                    hydra.kernels + i * self.kernel_size,
                    self.kernel_size,
                    &X[sample, feature.dim, 0],
                    X.shape[2],
                    self.conv_values + i * X.shape[2],
                )

            memset(self.min_values, 0, sizeof(double) * self.n_kernels)
            memset(self.max_values, 0, sizeof(double) * self.n_kernels)

            # Find the timestep-wise kernel with the minimum and maximum
            # values. We iterate self.conv_values with a stride equal
            # to the number of timesteps.
            for i in range(X.shape[2]):
                find_min_max(
                    i,
                    X.shape[2],
                    self.conv_values, 
                    self.n_kernels,
                    &min_index, 
                    &min_value, 
                    &max_index, 
                    &max_value
                )

                # NOTE: min_index and max_index are bounded by self.n_kernels
                self.min_values[min_index] += 1
                self.max_values[max_index] += max_value
           
            # We allocate each feature to an array with the following layout,
            # self.n_kernels * self.max_exponent * 2:
            #
            #        d=0            d=1
            # --------------- ---------------
            #   k=0     k=1     k=1     k=1
            # ------- ------- ------- -------
            # f=0 f=1 f=0 f=1 f=0 f=1 f=0 f=1
            # --- --- --- --- --- --- --- ---
            #
            # With self.n_groups such groups (one for each feature_id)
            #
            # Here we move the pointer to first kernel of the d:th dilation
            # making sure that we account for the fact that each kernel
            # is descrived by two features.
            kernel_feature_offset = feature_offset + exponent * self.n_kernels * 2 
            for i in range(self.n_kernels):
                # NOTE: *2 is the number of features (min/max)
                out[out_sample, kernel_feature_offset + i * 2] = self.min_values[i]
                out[out_sample, kernel_feature_offset + i * 2 + 1] = self.max_values[i]
            
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

cdef void find_min_max(
    Py_ssize_t offset,
    Py_ssize_t stride,
    double* values,
    Py_ssize_t length, 
    Py_ssize_t *min_index,
    double *min_value,
    Py_ssize_t *max_index,
    double *max_value
) noexcept nogil:
    cdef Py_ssize_t i
    cdef double value
    min_index[0] = -1
    max_index[0] = -1
    max_value[0] = -INFINITY
    min_value[0] = INFINITY

    for i in range(length):
        value = values[offset + i * stride]
        if value > max_value[0]:
            max_value[0] = value
            max_index[0] = i

        if value < min_value[0]:
            min_value[0] = value
            min_index[0] = i
