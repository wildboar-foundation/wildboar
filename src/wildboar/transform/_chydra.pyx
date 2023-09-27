# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

import numpy as np

from libc.math cimport INFINITY, floor, log2, pow, sqrt, fabs
from libc.stdlib cimport free, malloc, labs
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
        cdef Py_ssize_t max_exponent = min(
            1,
            <Py_ssize_t> floor(log2((X.shape[2] - 1) / (self.kernel_size - 1)))
        )
        cdef Hydra *hydra = <Hydra*> feature.feature
        cdef double *kernel 
        cdef double mean_val

        # Hydra (2023), D (transform, Line 5)
        for dilation in range(max_exponent + 1):
            padding = <Py_ssize_t> ((self.kernel_size - 1) * pow(2, dilation)) // 2

            for i in range(self.n_kernels):
                kernel = hydra.kernels + i * self.kernel_size
                convolution_1d(
                    1,
                    dilation,
                    padding,
                    1.0,
                    kernel,
                    self.kernel_size,
                    &X[sample, feature.dim, 0],
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

cdef void convolution_1d(
    Py_ssize_t stride,
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double bias,
    double *kernel,
    Py_ssize_t k_len,
    const double* x,
    Py_ssize_t x_len,
    double* out,
) noexcept nogil:
    """
    Compute the 1d convolution.

    Parameters
    ----------
    stride : int
        The stride in x.
    dilation : int
        Inflate the kernel by inserting spaces between kernel values.
    padding : int
        Increase the size of the input by padding.
    bias : double
        The bias.
    kernel : double*
        The kernel values.
    k_len : int
        The length of the kernel.
    x : double*
        The input sample.
    x_len : int
        The length of the sample.
    out : double*, output
        The output buffer. The length of `out` must be at least:
            ((x_len + 2 * padding) - (k_len - 1) * dilation + 1)
            ---------------------------------------------------- + 1
                                stride

        The invariant is not checked.
    """
    cdef Py_ssize_t input_size = x_len + 2 * padding
    cdef Py_ssize_t kernel_size = (k_len - 1) * dilation + 1
    cdef Py_ssize_t output_size = <Py_ssize_t> floor((input_size - kernel_size) / stride) + 1

    cdef Py_ssize_t j  # the index in the kernel and input array
    cdef Py_ssize_t i  # the index of the output array
    cdef Py_ssize_t padding_offset
    cdef Py_ssize_t input_offset
    cdef Py_ssize_t kernel_offset
    cdef Py_ssize_t convolution_size
    cdef double inner_prod

    for i in range(output_size):
        padding_offset = padding - i * stride
        
        # This part of the code ensures that the iterators responsible
        # for selecting the correct values in `kernel` and `x` start
        # at the correct location. The main idea is to always start
        # the index at the first non-dilated index *after* padding.
        #
        # Example:
        #  k = [1, 1, 2]
        #  d = 2
        #  k_d = [1, 0, 1, 0, 2]
        #  x = [1, 2, 3, 4, 5, 6]
        #  p = 2
        #  x_p = [0, 0, 1, 2, 3, 4, 5, 6, 0, 0]
        # 
        #  First convoluton:
        #          
        #          / start the kernel index here, which is given by
        #          | since 2 % 2 == 0, which is given by padding_offset
        #          |
        # k_p  1 0 1 0 2
        #      | | | | |
        # x_p  0 0 1 2 3 4 5 6 0 0 (Result: 7)
        #      |   |
        #      |   \ Start the input index here, which is given by
        #      |     kernel_offset - padding_offset (2 - 2 = 0)            
        #      |
        #      \ padding_offset is 2, so we are at the first padded value
        #        but we can ignore this part since the pad is all zeros
        #        so we move head of the kernel iterator to first non-dilated
        #        value, which is located at imaginary index 2, which also
        #        happens to be at the start of the input.
        # 
        #  Second convolution
        # 
        #
        #            / Since 1 % 2 != 0, we move the start of the kernel to the
        #            | the first non-dilated value, which is given by
        #            | 1 + 2 - (1 % 2) = 2
        #            |
        # k_p    1 0 1 0 2
        #        | | | | |
        # x_p  0 0 1 2 3 4 5 6 0 0 (Result: 10)
        #        | |
        #        | \ We move the input iterator the the index where the
        #        |   first non dilated value is, kernel_offset (2) - padding_offset (1)
        #        \ 
        #         padding_offset = 1, so we are at the second padded value.
        #
        #  Third convolution
        #
        #          / padding_offset = 0, so we should start the kernel
        #          | with the first value (which by definition is non-dilated).
        #          |
        # k_p      1 0 1 0 2
        #          | | | | |
        # x_p  0 0 1 2 3 4 5 6 0 0 (Result: 14)
        #          |
        #          \ we should start x at the current (strided) index.
        # 
        # We continue iterating, until we have visited all start locations
        # but stopping the convolution at the end of the input array.
        if padding_offset > 0:
            if padding_offset % dilation == 0:
                kernel_offset = padding_offset
            else:
                kernel_offset = padding_offset + dilation - (padding_offset % dilation) 
            input_offset = kernel_offset - padding_offset
        else:
            kernel_offset = 0
            input_offset = labs(padding_offset)
            
        # The iteration should be performed up until but not including the
        # padding. So, the last value we should convolve over is either the
        # last value of the input, or a value before that located where the
        # kernel ends. The kernel size here takes into account dilation. We
        # express this as a length from input_offset until the end.
        convolution_size = (
            min(x_len, input_offset + kernel_size - max(0, padding_offset))
            - input_offset
        )
        inner_prod = bias
        for j from 0 <= j < convolution_size by dilation:
            inner_prod += x[input_offset + j] * kernel[((j + kernel_offset) // dilation)]
            
        out[i] = inner_prod
