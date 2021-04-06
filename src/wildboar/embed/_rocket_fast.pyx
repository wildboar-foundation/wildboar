# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied wweightsanty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
cimport numpy as np
import numpy as np

from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libc.string cimport memcpy

from libc.math cimport INFINITY
from libc.math cimport sqrt
from libc.math cimport log2
from libc.math cimport pow
from libc.math cimport floor


from .._data cimport TSDatabase
from .._data cimport ts_database_new

from .._utils cimport rand_normal
from .._utils cimport rand_uniform
from .._utils cimport rand_int
from .._utils cimport RAND_R_MAX

from ._feature cimport FeatureEngineer
from ._feature cimport Feature

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

cdef class RocketFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t n_kernels
    cdef Py_ssize_t *lengths
    cdef Py_ssize_t n_lengths

    def __cinit__(self, n_kernels, lengths):
        self.n_kernels = n_kernels
        self.lengths = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * len(lengths))
        self.n_lengths = len(lengths)

        cdef Py_ssize_t i
        for i in range(len(lengths)):
            self.lengths[i] = lengths[i]

    def __reduce__(self):
        return self.__class__, (self.n_kernels, _to_ndarray(self.lengths, self.n_lengths))

    def __dealloc__(self):
        free(self.lengths)

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
        rocket_init(rocket, td.n_timestep, self.lengths, self.n_lengths, seed)
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
        rocket_apply(
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
        rocket_apply(
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

        return (feature.dim, (rocket.length, rocket.dilation, rocket.padding, weights, rocket.bias, rocket.return_mean))

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


cdef void rocket_init(
    Rocket *rocket,
    Py_ssize_t n_timestep,
    Py_ssize_t *lengths,
    Py_ssize_t n_lengths,
    size_t *random_state) nogil:
    cdef Py_ssize_t i
    cdef double mean = 0.0
    cdef Py_ssize_t length = lengths[rand_int(0, n_lengths, random_state)]
    cdef double* weight = <double*> malloc(sizeof(double) * length)
    
    for i in range(length):
        weight[i] = rand_normal(0, 1, random_state)
        mean += weight[i]
    mean = mean / length
    for i in range(length):
        weight[i] -= mean
    
    rocket.length = length
    rocket.dilation = <Py_ssize_t> floor(pow(2, rand_uniform(0, log2((n_timestep - 1) / (rocket.length - 1)), random_state)))
    rocket.padding = 0
    if rand_uniform(0, 1, random_state) < 0.5:
        rocket.padding = ((rocket.length - 1) * rocket.dilation) // 2
    rocket.return_mean = rand_uniform(0, 1, random_state) < 0.5
    rocket.weight = weight
    rocket.bias = rand_uniform(-1, 1, random_state)


cdef void rocket_apply(
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