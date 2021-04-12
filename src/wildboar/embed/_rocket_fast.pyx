# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten

cimport numpy as np

import numpy as np

from libc.math cimport INFINITY, floor, log2, pow, sqrt
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

from .._data cimport TSDatabase, ts_database_new
from .._utils cimport RAND_R_MAX, rand_int, rand_normal, rand_uniform
from ._feature cimport Feature, FeatureEngineer


cdef struct Rocket:
    Py_ssize_t length
    Py_ssize_t dilation
    Py_ssize_t padding
    bint return_mean
    double bias
    double *weight

cdef np.ndarray _to_ndarray(Py_ssize_t *arr, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef np.ndarray out = np.zeros(n, dtype=int)
    for i in range(n):
        out[i] = arr[i]

    return out

cdef class WeightSampler:

    cdef void sample(
        self,
        TSDatabase *td,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double *weights,
        Py_ssize_t length,
        double *mean,
        size_t *seed
    ) nogil:
        pass

cdef class NormalWeightSampler(WeightSampler):
    cdef double mean
    cdef double scale

    def __init__(self, mean=0.0, scale=1.0):
        self.mean = mean
        self.scale = scale

    cdef void sample(
        self,
        TSDatabase *td,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double *weights,
        Py_ssize_t length,
        double *mean,
        size_t *seed
    ) nogil:
        cdef Py_ssize_t i
        mean[0] = 0
        for i in range(length):
            weights[i] = rand_normal(self.mean, self.scale, seed)
            mean[0] += weights[i]
        mean[0] = mean[0] / length

cdef class UniformWeightSampler(WeightSampler):

    cdef double lower
    cdef double upper

    def __init__(self, lower=-1.0, upper=1.0):
        self.lower = lower
        self.upper = upper

    cdef void sample(
        self,
        TSDatabase *td,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double *weights,
        Py_ssize_t length,
        double *mean,
        size_t *seed
    ) nogil:
        cdef Py_ssize_t i
        mean[0] = 0
        for i in range(length):
            weights[i] = rand_uniform(self.lower, self.upper, seed)
            mean[0] += weights[i]
        mean[0] = mean[0] / length

cdef class ShapeletWeightSampler(WeightSampler):
    cdef void sample(
        self,
        TSDatabase *td,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double *weights,
        Py_ssize_t length,
        double *mean,
        size_t *seed
    ) nogil:
        cdef Py_ssize_t start
        cdef Py_ssize_t index
        cdef Py_ssize_t dim
        cdef Py_ssize_t i, offset

        start = rand_int(0, td.n_timestep - length, seed)
        index = samples[rand_int(0, n_samples, seed)]
        if td.n_dims > 1:
            dim = rand_int(0, td.n_dims, seed)
        else:
            dim = 1

        offset = (
            index * td.sample_stride +
            start * td.timestep_stride +
            dim * td.dim_stride
        )
        mean[0] = 0
        for i in range(length):
            weights[i] = td.data[offset + i * td.timestep_stride]
            mean[0] += weights[i]
        mean[0] /= length


cdef class RocketFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t n_kernels
    cdef WeightSampler weight_sampler
    cdef double padding_prob
    cdef double bias_prob
    cdef double normalize_prob
    cdef Py_ssize_t *kernel_size
    cdef Py_ssize_t n_kernel_size

    def __cinit__(
        self,
        Py_ssize_t n_kernels,
        WeightSampler weight_sampler,
        np.ndarray kernel_size,
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
            _to_ndarray(self.kernel_size, self.n_kernel_size),
            self.bias_prob,
            self.padding_prob,
            self.normalize_prob
        )

    def __dealloc__(self):
        free(self.kernel_size)

    cdef Py_ssize_t get_n_features(self, TSDatabase *td) nogil:
        return self.n_kernels

    cdef Py_ssize_t get_n_outputs(self, TSDatabase *td) nogil:
        return self.get_n_features(td) * 2

    cdef Py_ssize_t next_feature(
        self,
        Py_ssize_t feature_id,
        TSDatabase *td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        Feature *transient,
        size_t *seed
    ) nogil:
        cdef Rocket *rocket = <Rocket*> malloc(sizeof(Rocket))
        cdef Py_ssize_t i
        cdef double mean
        cdef Py_ssize_t length = self.kernel_size[rand_int(0, self.n_kernel_size, seed)]
        cdef double* weight = <double*> malloc(sizeof(double) * length)

        self.weight_sampler.sample(
            td, samples, n_samples, weight, length, &mean, seed
        )
        if rand_uniform(0, 1, seed) < self.normalize_prob:
            for i in range(length):
                weight[i] -= mean

        rocket.length = length
        rocket.dilation = <Py_ssize_t> floor(
            pow(2, rand_uniform(0, log2((td.n_timestep - 1) / (rocket.length - 1)), seed))
        )
        rocket.padding = 0
        if rand_uniform(0, 1, seed) < self.padding_prob:
            rocket.padding = ((rocket.length - 1) * rocket.dilation) // 2

        rocket.return_mean = rand_uniform(0, 1, seed) < 0.5
        rocket.weight = weight

        rocket.bias = 0
        if rand_uniform(0, 1, seed) < self.bias_prob:
            rocket.bias = rand_uniform(-1, 1, seed)

        transient.dim = 1
        transient.feature = rocket
        return 0

    cdef Py_ssize_t free_transient_feature(self, Feature *feature) nogil:
        cdef Rocket *rocket
        if feature.feature != NULL:
            rocket = <Rocket*> feature.feature
            if rocket.weight != NULL:
                free(rocket.weight)
            free(feature.feature)
        return 0

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) nogil:
        return self.free_transient_feature(feature)

    cdef Py_ssize_t init_persistent_feature(
        self, 
        TSDatabase *td,
        Feature *transient, 
        Feature *persistent
    ) nogil:
        cdef Rocket *transient_rocket = <Rocket*> transient.feature
        cdef Rocket *persistent_rocket = <Rocket*> malloc(sizeof(Rocket))
        
        persistent_rocket.dilation = transient_rocket.dilation
        persistent_rocket.bias = transient_rocket.bias
        persistent_rocket.length = transient_rocket.length
        persistent_rocket.padding = transient_rocket.padding
        persistent_rocket.return_mean = transient_rocket.return_mean
        persistent_rocket.weight = <double*> malloc(sizeof(double) * transient_rocket.length)
        memcpy(
            persistent_rocket.weight, 
            transient_rocket.weight, 
            sizeof(double) * transient_rocket.length,
        )
        persistent.dim = transient.dim
        persistent.feature = persistent_rocket
        return 0

    cdef double transient_feature_value(
        self,
        Feature *feature,
        TSDatabase *td,
        Py_ssize_t sample
    ) nogil:
        cdef double mean_val, max_val
        cdef Py_ssize_t sample_offset = (
            sample * td.sample_stride + feature.dim * td.dim_stride
        )
        cdef Rocket* rocket = <Rocket*> feature.feature
        apply_convolution(
            rocket.length,
            rocket.dilation,
            rocket.padding,
            rocket.bias,
            0,
            1,
            rocket.weight,
            sample_offset, 
            td.timestep_stride,
            td.n_timestep,
            td.data,
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
        TSDatabase *td,
        Py_ssize_t sample
    ) nogil:
        return self.transient_feature_value(feature, td, sample)

    cdef Py_ssize_t transient_feature_fill(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t sample,
        TSDatabase *td_out,
        Py_ssize_t out_sample,
        Py_ssize_t feature_id,
    ) nogil:
        cdef double mean_val, max_val
        cdef Py_ssize_t sample_offset = (
            sample * td.sample_stride + feature.dim * td.dim_stride
        )
        cdef Rocket* rocket = <Rocket*> feature.feature
        apply_convolution(
            rocket.length,
            rocket.dilation,
            rocket.padding,
            rocket.bias,
            0,
            1,
            rocket.weight,
            sample_offset,
            td.timestep_stride,
            td.n_timestep,
            td.data,
            &mean_val,
            &max_val,
        )
        cdef Py_ssize_t feature_offset = feature_id * 2 * td_out.timestep_stride
        cdef Py_ssize_t out_sample_offset = out_sample * td_out.sample_stride
        td_out.data[out_sample_offset + feature_offset] = mean_val
        td_out.data[out_sample_offset + feature_offset + td_out.timestep_stride] = max_val
        return 0

    cdef Py_ssize_t persistent_feature_fill(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t sample,
        TSDatabase *td_out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) nogil:
        return self.transient_feature_fill(
            feature, td, sample, td_out, out_sample, out_feature
        )

    cdef object persistent_feature_to_object(self, Feature *feature):
        cdef Py_ssize_t j
        cdef Rocket *rocket = <Rocket*> feature.feature

        weights = np.empty(rocket.length, dtype=np.float64)
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

cdef void apply_convolution(
    Py_ssize_t length,
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double bias,
    Py_ssize_t w_offset,
    Py_ssize_t w_stride,
    double *weight,
    Py_ssize_t x_offset,
    Py_ssize_t x_stride,
    Py_ssize_t x_length,
    double* x,
    double* mean_val,
    double* max_val
) nogil:
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
                inner_prod += weight[w_offset + w_stride * j] * x[x_offset + x_stride * k]
            k += dilation
        if inner_prod > max_val[0]:
            max_val[0] = inner_prod

        if inner_prod > 0:
            mean_val[0] += 1

    mean_val[0] /= out_len