# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# This file is part of pypf
#
# pypf is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pypf is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

# Authors: Isak Karlsson

import numpy as np
cimport numpy as np

cimport cython

from libc.stdlib cimport realloc
from libc.stdlib cimport malloc
from libc.stdlib cimport free

from libc.math cimport sqrt
from libc.math cimport INFINITY, NAN

from sklearn.utils import check_array


cdef class DistanceMeasure:

    cdef void init(self, TSDatabase td) nogil:
        self.td = td

    cdef void distances(self, ShapeletInfo s, size_t* samples,
                        double* distances, size_t n_samples) nogil:
        cdef size_t p
        for p in range(n_samples):
            distances[p] = self.distance(s, samples[p])

    cdef ShapeletInfo new_shapelet_info(self,
                                        size_t index,
                                        size_t dim,
                                        size_t start,
                                        size_t length) nogil:
        cdef ShapeletInfo shapelet_info
        shapelet_info.index = index
        shapelet_info.dim = dim
        shapelet_info.start = start
        shapelet_info.length = length
        shapelet_info.mean = NAN
        shapelet_info.std = NAN
        return shapelet_info

    cdef Shapelet new_shapelet(self, ShapeletInfo s):
        raise NotImplemented()

    cdef double distance(self, ShapeletInfo s, size_t t_index) nogil:
        with gil:
            raise NotImplemented()


cdef class ScaledDistanceMeasure(DistanceMeasure):

    cdef ShapeletInfo new_shapelet_info(self, size_t index, size_t dim,
                                        size_t start, size_t length) nogil:
        cdef ShapeletInfo shapelet_info
        shapelet_info.index = index
        shapelet_info.dim = dim
        shapelet_info.start = start
        shapelet_info.length = length
        shapelet_info_update_statistics(&shapelet_info, self.td)
        return shapelet_info


cdef class ScaledEuclideanDistance(ScaledDistanceMeasure):

    cdef Shapelet new_shapelet(self, ShapeletInfo s):
        cdef Shapelet shapelet = ScaledShapelet(s.dim, s.length, s.mean, s.std)
        cdef double* data = shapelet.data
        cdef size_t shapelet_offset = (s.index * self.td.sample_stride +
                                       s.start * self.td.timestep_stride +
                                       s.dim * self.td.dim_stride)

        cdef size_t i
        cdef size_t p
        with nogil:
            for i in range(s.length):
                p = shapelet_offset + self.td.timestep_stride * i
                data[i] = self.td.data[p]
        return shapelet

    cdef double distance(self, ShapeletInfo s, size_t t_index) nogil:
        cdef size_t sample_offset = (t_index * self.td.sample_stride + 
                                     s.dim * self.td.dim_stride)
        cdef size_t shapelet_offset = (s.index * self.td.sample_stride +
                                       s.dim * self.td.dim_stride + 
                                       s.start * self.td.timestep_stride)
        return scaled_euclidean_distance(
            shapelet_offset,
            self.td.timestep_stride,
            s.length,
            s.mean,
            s.std,
            self.td.data,
            sample_offset,
            self.td.timestep_stride,
            self.td.n_timestep,
            self.td.data,
            self.td.X_buffer,
            NULL)


cdef class EuclideanDistance(DistanceMeasure):

    cdef Shapelet new_shapelet(self, ShapeletInfo s):
        cdef Shapelet shapelet = Shapelet(s.dim, s.length)
        cdef double* data = shapelet.data
        cdef size_t shapelet_offset = (s.index * self.td.sample_stride +
                                       s.start * self.td.timestep_stride +
                                       s.dim * self.td.dim_stride)
        cdef size_t i
        cdef size_t p
        with nogil:
            for i in range(s.length):
                p = shapelet_offset + self.td.timestep_stride * i
                data[i] = self.td.data[p]

        return shapelet

    cdef double distance(self, ShapeletInfo s, size_t t_index) nogil:
        cdef size_t sample_offset = (t_index * self.td.sample_stride + 
                                     s.dim * self.td.dim_stride)
        cdef size_t shapelet_offset = (s.index * self.td.sample_stride +
                                       s.dim * self.td.dim_stride + 
                                       s.start * self.td.timestep_stride)
        return euclidean_distance(
            shapelet_offset,
            self.td.timestep_stride,
            s.length,
            self.td.data,
            sample_offset,
            self.td.timestep_stride,
            self.td.n_timestep,
            self.td.data,
            NULL)    


cpdef Shapelet make_scaled_shapelet_(size_t dim, size_t length, double mean,
                                     double std, object array):
    """Reconstruct a `Shapelet`-object from Pickle

    :param length: the size of the shapelet
    :param array: the Numpy array
    :return: a shapelet
    """
    cdef Shapelet shapelet = ScaledShapelet(dim, length, mean, std)
    cdef size_t i
    for i in range(<size_t> array.shape[0]):
        shapelet.data[i] = array[i]

    return shapelet


cpdef Shapelet make_shapelet_(size_t dim, size_t length, object array):
    """Reconstruct a `Shapelet`-object from Pickle

    :param length: the size of the shapelet
    :param array: the Numpy array
    :return: a shapelet
    """
    cdef Shapelet shapelet = Shapelet(dim, length)
    cdef size_t i
    for i in range(<size_t> array.shape[0]):
        shapelet.data[i] = array[i]

    return shapelet


cdef class Shapelet:

    def __cinit__(self, size_t dim, size_t length, *args, **kvargs):
        self.length = length
        self.dim = dim
        self.data = <double*> malloc(sizeof(double) * length)
        if self.data == NULL:
            raise MemoryError()

    def __dealloc__(self):
        free(self.data)

    def __reduce__(self):
        return make_shapelet_, (self.dim, self.length, self.array)

    @property
    def array(self):
        cdef np.ndarray[np.float64_t] arr = np.empty(
            self.length, dtype=np.float64)
        cdef size_t i
        for i in range(self.length):
            arr[i] = self.data[i]
        return arr

    cdef double distance(self, const TSDatabase t, size_t t_index) nogil:
        cdef size_t sample_offset = (t_index * t.sample_stride +
                                     self.dim * t.dim_stride)
        return euclidean_distance(
            0,
            1,
            self.length,
            self.data,
            sample_offset,
            t.timestep_stride,
            t.n_timestep,
            t.data,
            NULL)

    cdef void distances(self,
                        const TSDatabase t,
                        size_t* samples,
                        size_t n_samples,
                        double* distances) nogil:
        cdef size_t i
        for i in range(n_samples):
            distances[i] = self.distance(t, samples[i])


cdef class ScaledShapelet(Shapelet):
    """Representing a shapelet carrying its own data.

    Note that the data is normalized during extraction if
    `shapelet_info_extract_shapelet` is used.
    """

    def __init__(self, size_t dim, size_t length, double mean, double std):
        """Initializes a shapelet with an empty c-array `data`.

        :param size_t length: the size of the shapelet
        """
        self.mean = mean
        self.std = std

    def __reduce__(self):
        return make_scaled_shapelet_, (self.dim, self.length,
                                       self.mean, self.std,
                                       self.array)

    cdef double distance(self, const TSDatabase t, size_t t_index) nogil:
        cdef size_t sample_offset = (t_index * t.sample_stride +
                                     self.dim * t.dim_stride)
        return scaled_euclidean_distance(
            0,
            1,
            self.length,
            self.mean,
            self.std,
            self.data,
            sample_offset,
            t.timestep_stride,
            t.n_timestep,
            t.data,
            t.X_buffer,
            NULL)


cdef int shapelet_info_update_statistics(ShapeletInfo* s,
                                         const TSDatabase t) nogil:
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)
    cdef double ex = 0
    cdef double ex2 = 0
    cdef size_t i
    for i in range(s.length):
        current_value = t.data[shapelet_offset + i * t.timestep_stride]
        ex += current_value
        ex2 += current_value**2

    s[0].mean = ex / s.length
    s[0].std = sqrt(ex2 / s.length - s[0].mean * s[0].mean)
    return 0


cdef TSDatabase new_ts_database(np.ndarray data):
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError("ndim {0} < 2 or {0} > 3".format(data.ndim))

    cdef TSDatabase sd
    sd.n_samples = <size_t> data.shape[0]
    sd.n_timestep = <size_t> data.shape[data.ndim - 1]
    sd.data = <double*> data.data
    sd.sample_stride = <size_t> data.strides[0] / <size_t> data.itemsize
    sd.timestep_stride = <size_t> data.strides[data.ndim - 1] / <size_t> data.itemsize

    if data.ndim == 3:
        sd.n_dims = <size_t> data.shape[data.ndim - 2]
        sd.dim_stride = <size_t> data.strides[data.ndim - 2] / <size_t> data.itemsize
    else:
        sd.n_dims = 1
        sd.dim_stride = 0

    sd.X_buffer = <double*> malloc(sizeof(double) * 2 * sd.n_timestep)

    if sd.X_buffer == NULL:
        raise MemoryError("error allocating buffer")

    return sd


cdef int free_ts_database(TSDatabase sd) nogil:
    free(sd.X_buffer)
    sd.X_buffer = NULL
    return 0


cdef double scaled_euclidean_distance(size_t s_offset,
                                      size_t s_stride,
                                      size_t s_length,
                                      double s_mean,
                                      double s_std,
                                      double* S,
                                      size_t t_offset,
                                      size_t t_stride,
                                      size_t t_length,
                                      double* T,
                                      double* X_buffer,
                                      size_t* index) nogil:
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef double ex = 0
    cdef double ex2 = 0

    cdef size_t i
    cdef size_t j
    cdef size_t buffer_pos

    for i in range(t_length):
        current_value = T[t_offset + t_stride * i]
        ex += current_value
        ex2 += current_value * current_value

        buffer_pos = i % s_length
        X_buffer[buffer_pos] = current_value
        X_buffer[buffer_pos + s_length] = current_value
        if i >= s_length - 1:
            j = (i + 1) % s_length
            mean = ex / s_length
            std = sqrt(ex2 / s_length - mean * mean)
            dist = inner_scaled_euclidean_distance(s_offset, s_length, s_mean, s_std,
                                                   j, mean, std, S, s_stride,
                                                   X_buffer, min_dist)

            if dist < min_dist:
                min_dist = dist
                if index != NULL:
                    index[0] = (i + 1) - s_length

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return sqrt(min_dist)


cdef inline double inner_scaled_euclidean_distance(size_t offset,
                                                   size_t length,
                                                   double s_mean,
                                                   double s_std,
                                                   size_t j,
                                                   double mean,
                                                   double std,
                                                   double* X,
                                                   size_t timestep_stride,
                                                   double* X_buffer,
                                                   double min_dist,
                                                   bint only_gt=False) nogil:
    # Compute the distance between the shapelet (starting at `offset`
    # and ending at `offset + length` normalized with `s_mean` and
    # `s_std` with the shapelet in `X_buffer` starting at `0` and
    # ending at `length` normalized with `mean` and `std`
    cdef double dist = 0
    cdef double x
    cdef size_t i
    cdef bint std_zero = std == 0
    cdef bint s_std_zero = s_std == 0

    # distance is zero
    if s_std_zero and std_zero:
        return 0

    for i in range(length):
        if only_gt:
            if dist > min_dist:
                break
        else:
            if dist >= min_dist:
                break

        x = (X[offset + timestep_stride * i] - s_mean) / s_std
        if not std_zero:
            x -= (X_buffer[i + j] - mean) / std
        dist += x * x

    return dist


cdef double euclidean_distance(size_t s_offset,
                               size_t s_stride,
                               size_t s_length,
                               double* S,
                               size_t t_offset,
                               size_t t_stride,
                               size_t t_length,
                               double* T,
                               size_t* index) nogil:
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef size_t i
    cdef size_t j
    cdef double x
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist >= min_dist:
                 break

            x = T[t_offset + t_stride * i + j]
            x -= S[s_offset + s_stride * j]
            dist += x * x

        if dist < min_dist:
            min_dist = dist
            if index != NULL:
                index[0] = i

    return sqrt(min_dist)


cdef int euclidean_distance_matches(size_t s_offset,
                                    size_t s_stride,
                                    size_t s_length,
                                    double* S,
                                    size_t t_offset,
                                    size_t t_stride,
                                    size_t t_length,
                                    double* T,
                                    double threshold,
                                    size_t** matches,
                                    size_t* n_matches) nogil except -1:
    cdef double dist = 0
    cdef size_t capacity = 1
    cdef size_t i
    cdef size_t j
    cdef double x

    matches[0] = <size_t*> malloc(sizeof(size_t) * capacity)
    n_matches[0] = 0

    threshold = threshold * threshold
    for i in range(t_length - s_length + 1):
        dist = 0
        for j in range(s_length):
            if dist > threshold:
                 break

            x = T[t_offset + t_stride * i + j]
            x -= S[s_offset + s_stride * j]
            dist += x * x
        if dist <= threshold:
            safe_add_to_array(matches, n_matches[0], i, &capacity)
            n_matches[0] += 1

    return 0


cdef double scaled_euclidean_distance_matches(size_t s_offset,
                                              size_t s_stride,
                                              size_t s_length,
                                              double s_mean,
                                              double s_std,
                                              double* S,
                                              size_t t_offset,
                                              size_t t_stride,
                                              size_t t_length,
                                              double* T,
                                              double* X_buffer,
                                              double threshold,
                                              size_t** matches,
                                              size_t* n_matches) nogil except -1:
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0

    cdef double ex = 0
    cdef double ex2 = 0

    cdef size_t i
    cdef size_t j
    cdef size_t buffer_pos
    cdef size_t capacity = 4

    matches[0] = <size_t*>malloc(sizeof(size_t) * capacity)
    n_matches[0] = 0

    threshold = threshold * threshold

    for i in range(t_length):
        current_value = T[t_offset + t_stride * i]
        ex += current_value
        ex2 += current_value * current_value

        buffer_pos = i % s_length
        X_buffer[buffer_pos] = current_value
        X_buffer[buffer_pos + s_length] = current_value
        if i >= s_length - 1:
            j = (i + 1) % s_length
            mean = ex / s_length
            std = sqrt(ex2 / s_length - mean * mean)
            dist = inner_scaled_euclidean_distance(s_offset, s_length, s_mean, s_std,
                                                   j, mean, std, S, s_stride,
                                                   X_buffer, threshold,
                                                   only_gt=False)
            if dist - threshold <= 1e-7: # TODO: improve
                safe_add_to_array(
                    matches,
                    n_matches[0],
                    (i + 1) - s_length,
                    &capacity)
                n_matches[0] += 1

            current_value = X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return 0


cdef int safe_add_to_array(size_t** a, size_t p,
                           size_t v, size_t* cap)  nogil except -1:
    cdef size_t* tmp = a[0]
    if p >= cap[0]:
        cap[0] *= 2
        tmp = <size_t*> realloc(a[0], sizeof(size_t) * cap[0])
        if tmp == NULL:
            with gil:
                raise MemoryError()

    a[0] = tmp
    a[0][p] = v
