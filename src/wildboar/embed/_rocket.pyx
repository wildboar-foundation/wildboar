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


# TODO: rename
cdef struct Rocket:
    Py_ssize_t length
    Py_ssize_t dilation
    Py_ssize_t padding
    bint return_mean
    double bias
    double *weight


# TODO: add to Rocket
cpdef enum RocketValue:
    MEAN = 1,
    MAX = 2,
    MIN = 3


cdef class RocketFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t n_kernels
    cdef size_t random_seed

    def __init__(self, n_kernels, random_state):
        self.n_kernels = n_kernels
        self.random_seed = random_state.randint(0, RAND_R_MAX)

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
        Feature *transient
    ) nogil:
        cdef Rocket *rocket = <Rocket*> malloc(sizeof(Rocket))
        rocket_init(rocket, td.n_timestep, &self.random_seed)
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
            rocket, 
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
            rocket, 
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


cdef void rocket_init(Rocket *rocket, Py_ssize_t n_timestep, size_t *random_state) nogil:
    cdef Py_ssize_t i
    cdef double mean = 0.0
    cdef Py_ssize_t length = 7
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


cdef void rocket_apply(Rocket *rocket, Py_ssize_t offset, Py_ssize_t stride, Py_ssize_t length, double* T, double* mean_val, double* max_val) nogil:
    cdef Py_ssize_t out_len, end
    cdef Py_ssize_t i, j, k
    cdef double ppv
    cdef double max
    cdef double sum

    out_len = (length + 2 * rocket.padding) - ((rocket.length - 1) * rocket.dilation)
    end = (length + rocket.padding) - ((rocket.length - 1) * rocket.dilation)
    max_val[0] = -INFINITY
    mean_val[0] = 0.0
    for i in range(-rocket.padding, end):
        sum = rocket.bias
        k = i
        for j in range(rocket.length):
            if k > -1 and k < length:
                sum += rocket.weight[j] * T[offset + stride * k]
            k += rocket.dilation
        if sum > max_val[0]:
            max_val[0] = sum

        if sum > 0:
            mean_val[0] += 1

    mean_val[0] /= out_len

def apply_kernel(X, weights, length, bias, dilation, padding):
    cdef double* V = <double*> malloc(sizeof(double) * len(X))
    cdef Py_ssize_t i
    for i in range(len(X)):
        V[i] = X[i]

    cdef Rocket rocket
    rocket.weight = <double*> malloc(sizeof(double) * length)
    for i in range(length):
        rocket.weight[i] = weights[i]

    rocket.bias = bias
    rocket.dilation = dilation
    rocket.padding = padding
    rocket.length = length
    cdef double mean_val, max_val
    rocket_apply(&rocket, 0, 1, len(X), V, &mean_val, &max_val)
    free(V)
    free(rocket.weight)
    return mean_val, max_val


# # TODO: rename
# cdef class RocketTransform:

#     cdef Py_ssize_t n_kernels
#     cdef object random_state
#     cdef Rocket** kernels

#     def __cinit__(self, *, n_kernels=10000, random_state=None):
#         self.n_kernels = n_kernels
#         self.random_state = random_state
#         self.kernels = <Rocket**> malloc(sizeof(Rocket) * n_kernels)

#     def __dealloc__(self):
#         for i in range(self.n_kernels):
#             free(self.kernels[i].weight)
#         free(self.kernels)

#     def fit(self, x, y=None):
#         random_state = check_random_state(self.random_state)
#         cdef size_t seed = random_state.randint(0, RAND_R_MAX)
#         cdef Rocket* rocket
#         cdef Py_ssize_t i
#         for i in range(self.n_kernels):
#             rocket = <Rocket*> malloc(sizeof(Rocket))
#             rocket_init(rocket, x.shape[1], &seed)
#             self.kernels[i] = rocket
#         return self

#     def transform(self, x, y=None):
#         x_out = np.empty((x.shape[0], self.n_kernels * 2))
#         cdef TSDatabase td = ts_database_new(x)
#         cdef Py_ssize_t sample_offset
#         cdef Rocket* rocket
#         cdef double mean_val = 0, max_val = 0
#         for i in range(x.shape[0]):
#             sample_offset = i * td.sample_stride
#             k = 0
#             for j in range(self.n_kernels):
#                 rocket = self.kernels[j]
#                 rocket_apply(rocket, sample_offset, td.timestep_stride, td.n_timestep, td.data, &mean_val, &max_val)
#                 x_out[i, k] = mean_val
#                 x_out[i, k + 1] = max_val
#                 k += 2
#         return x_out



