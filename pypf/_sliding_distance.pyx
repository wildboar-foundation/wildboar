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

from libc.stdlib cimport malloc
from libc.stdlib cimport free

from libc.math cimport sqrt
from libc.math cimport INFINITY

from sklearn.utils import check_array

from pypf._utils cimport checked_realloc

cpdef Shapelet _make_scaled_shapelet(size_t length, double mean, double std, object array):
    """Reconstruct a `Shapelet`-object from Pickle

    :param length: the size of the shapelet
    :param array: the Numpy array
    :return: a shapelet
    """
    cdef Shapelet shapelet = ScaledShapelet(length, mean, std)
    cdef size_t i
    for i in range(<size_t> array.shape[0]):
        shapelet.data[i] = array[i]

    return shapelet

cpdef Shapelet _make_shapelet(size_t length, object array):
    """Reconstruct a `Shapelet`-object from Pickle

    :param length: the size of the shapelet
    :param array: the Numpy array
    :return: a shapelet
    """
    cdef Shapelet shapelet = Shapelet(length)
    cdef size_t i
    for i in range(<size_t> array.shape[0]):
        shapelet.data[i] = array[i]

    return shapelet

cdef class Shapelet:

    def __cinit__(self, size_t length, *args, **kvargs):
        self.length = length
        self.data = <double*> malloc(sizeof(double) * length)
        if self.data == NULL:
            raise MemoryError()

    def __dealloc__(self):
        free(self.data)

    def __reduce__(self):
        return (_make_shapelet, (self.length, self.array))

    @property
    def array(self):
        cdef np.ndarray[np.float64_t] arr = np.empty(self.length,
                                                     dtype=np.float64)
        cdef size_t i
        for i in range(self.length):
            arr[i] = self.data[i]
        return arr

    @property
    def unscaled_array(self):
        return self.array

    cdef double distance(self, const SlidingDistance t, size_t t_index) nogil:
        cdef size_t sample_offset = t_index * t.sample_stride

        cdef double x
        cdef double dist = 0
        cdef double min_dist = INFINITY

        cdef size_t i
        cdef size_t j        
        for i in range(t.n_timestep - self.length + 1):
            dist = 0
            for j in range(self.length):
                # if dist >= min_dist:
                #     break

                x = t.X[sample_offset + t.timestep_stride * i + j]
                x -= self.data[j]
                dist += x * x
            if dist < min_dist:
                min_dist = dist

        return sqrt(min_dist)

    cdef void distances(self,
                       const SlidingDistance t,
                       size_t* samples,
                       size_t n_samples,
                       double* distances) nogil:
        cdef size_t i
        for i in range(n_samples):
            distances[i] = self.distance(t, samples[i])

    cdef size_t closer_than(self,
                           const SlidingDistance t,
                           size_t t_index,
                           double threshold,
                           size_t* matches,
                           double* distances,
                           size_t initial_capacity) nogil:
        cdef size_t sample_offset = t_index * t.sample_stride

        cdef double dist = 0
        cdef size_t min_index = 0

        cdef size_t i
        cdef size_t j

        cdef size_t m = 0
        cdef size_t capacity = initial_capacity

        # increase the threshold to avoid having to repeatedly call
        # `sqrt` before comparision to the threshold
        threshold = threshold ** 2

        for i in range(t.n_timestep):
            dist = 0
            for j in range(self.length):
                if dist >= threshold:
                    break

                x = t.X[sample_offset + t.timestep_stride * i + j]
                x -= self.data[j]
                dist += x * x
                
                if dist < threshold:
                    if m >= capacity:
                        # TODO: fixme
                        capacity *= 2
                        checked_realloc(
                            <void**> &matches, capacity * sizeof(size_t))
                        checked_realloc(
                            <void**> &distances, capacity * sizeof(double))

                    matches[m] = i
                    distances[m] = sqrt(dist)
                    m += 1
        return m

    cdef size_t index_distance(self,
                              const SlidingDistance t,
                              size_t t_index,
                              double* min_dist) nogil:
        cdef size_t sample_offset = t_index * t.sample_stride

        cdef double dist = 0
        
        cdef size_t min_index = 0
        min_dist[0] = INFINITY

        cdef size_t i
        cdef size_t j
        for i in range(t.n_timestep - self.length + 1):
            dist = 0
            for j in range(self.length):
                if dist >= min_dist[0]:
                    break

                x = t.X[sample_offset + t.timestep_stride * i + j]
                x -= self.data[j]
                dist += x * x
            if dist < min_dist[0]:
                min_dist[0] = dist
                min_index = i

        min_dist[0] = sqrt(min_dist[0])
        return min_index

    cdef void index_distances(self,
                             const SlidingDistance t,
                             size_t* samples,
                             size_t n_samples,
                             size_t* min_indicies,
                              double* min_distances) nogil:
        cdef size_t i
        cdef size_t min_index
        cdef double min_dist
        for i in range(n_samples):
            min_index = self.index_distance(t, samples[i], &min_dist)
            min_indicies[i] = min_index
            min_distances[i] = min_dist


cdef class ScaledShapelet(Shapelet):
    """Representing a shapelet carrying its own data.

    Note that the data is normalized during extraction if
    `shapelet_info_extract_shapelet` is used.
    """

    def __init__(self, size_t length, double mean, double std):
        """Initializes a shapelet with an empty c-array `data`.

        :param size_t length: the size of the shapelet
        """
        self.mean = mean
        self.std = std

    def __reduce__(self):
        return (_make_scaled_shapelet, (self.length, self.mean, self.std,
                                 self.array))

    @property
    def array(self):
        cdef np.ndarray[np.float64_t] arr = np.empty(self.length,
                                                     dtype=np.float64)
        cdef size_t i
        for i in range(self.length):
            arr[i] = self.data[i]
        return arr

    @property
    def unscaled_array(self):
        return self.array * self.std + self.mean

    cdef double distance(self, const SlidingDistance t, size_t t_index) nogil:
        cdef size_t sample_offset = t_index * t.sample_stride
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

        for i in range(t.n_timestep):
            current_value = t.X[sample_offset + t.timestep_stride * i]
            ex += current_value
            ex2 += current_value * current_value

            buffer_pos = i % self.length
            t.X_buffer[buffer_pos] = current_value
            t.X_buffer[buffer_pos + self.length] = current_value
            if i >= self.length - 1:
                j = (i + 1) % self.length
                mean = ex / self.length
                std = sqrt(ex2 / self.length - mean * mean)
                dist = shapelet_subsequence_distance(
                    self.length, # length of shapelet
                    self.data,   # normalized shapelet
                    j,           # buffer offset
                    mean,        # buffer mean
                    std,         # buffer std
                    t.X_buffer,
                    min_dist)

                if dist < min_dist:
                    min_dist = dist

                current_value = t.X_buffer[j]
                ex -= current_value
                ex2 -= current_value * current_value

        return sqrt(min_dist)

    cdef size_t index_distance(self,
                               const SlidingDistance t,
                               size_t t_index,
                               double* min_dist) nogil:
        cdef size_t sample_offset = t_index * t.sample_stride
        cdef double current_value = 0
        cdef double mean = 0
        cdef double std = 0
        cdef double dist = 0

        cdef size_t min_index = 0
        min_dist[0] = INFINITY

        cdef double ex = 0
        cdef double ex2 = 0

        cdef size_t i
        cdef size_t j
        cdef size_t buffer_pos

        for i in range(t.n_timestep):
            current_value = t.X[sample_offset + t.timestep_stride * i]
            ex += current_value
            ex2 += current_value * current_value

            buffer_pos = i % self.length
            t.X_buffer[buffer_pos] = current_value
            t.X_buffer[buffer_pos + self.length] = current_value
            if i >= self.length - 1:
                j = (i + 1) % self.length
                mean = ex / self.length
                std = sqrt(ex2 / self.length - mean * mean)
                dist = shapelet_subsequence_distance(
                    self.length, # length of shapelet
                    self.data,   # normalized shapelet
                    j,           # buffer offset
                    mean,        # buffer mean
                    std,         # buffer std
                    t.X_buffer,
                    min_dist[0])

                if dist < min_dist[0]:
                    min_dist[0] = dist
                    min_index = (i + 1) - self.length

                current_value = t.X_buffer[j]
                ex -= current_value
                ex2 -= current_value * current_value

        min_dist[0] = sqrt(min_dist[0])
        return min_index

    cdef size_t closer_than(self,
                            const SlidingDistance t,
                            size_t t_index,
                            double threshold,
                            size_t* matches,
                            double* distances,
                            size_t initial_capacity) nogil:
        cdef size_t sample_offset = t_index * t.sample_stride
        cdef double current_value = 0
        cdef double mean = 0
        cdef double std = 0
        cdef double dist = 0

        cdef size_t min_index = 0
        cdef double ex = 0
        cdef double ex2 = 0

        cdef size_t i
        cdef size_t j
        cdef size_t buffer_pos

        cdef size_t m = 0
        cdef size_t capacity = initial_capacity

        # increase the threshold to avoid having to repeatedly call
        # `sqrt` before comparision to the threshold
        threshold = threshold ** 2

        for i in range(t.n_timestep):
            current_value = t.X[sample_offset + t.timestep_stride * i]
            ex += current_value
            ex2 += current_value * current_value

            buffer_pos = i % self.length
            t.X_buffer[buffer_pos] = current_value
            t.X_buffer[buffer_pos + self.length] = current_value
            if i >= self.length - 1:
                j = (i + 1) % self.length
                mean = ex / self.length
                std = sqrt(ex2 / self.length - mean * mean)
                dist = shapelet_subsequence_distance(
                    self.length, # length of shapelet
                    self.data,   # normalized shapelet
                    j,           # buffer offset
                    mean,        # buffer mean
                    std,         # buffer std
                    t.X_buffer,
                    threshold)

                # <= to support threshold 0
                if dist < threshold:
                    if m >= capacity:
                        capacity *= 2
                        checked_realloc(
                            <void**> &matches, capacity * sizeof(size_t))
                        checked_realloc(
                            <void **> &distances, capacity * sizeof(double))

                    matches[m] = (i + 1) - self.length
                    distances[m] = sqrt(dist)
                    m += 1

                current_value = t.X_buffer[j]
                ex -= current_value
                ex2 -= current_value * current_value

        return m


cdef inline double shapelet_subsequence_distance(size_t length,
                                                 double* shapelet,
                                                 size_t j,
                                                 double mean,
                                                 double std,
                                                 double* X_buffer,
                                                 double min_dist) nogil:
    cdef double dist = 0
    cdef double x
    cdef size_t i
    cdef bint std_zero = std == 0
    for i in range(length):
        if dist >= min_dist:
            break

        x = shapelet[i]
        if not std_zero:
            x -= (X_buffer[i + j] - mean) / std
        dist += x * x

    return dist


cdef inline double shapelet_info_subsequence_distance(size_t offset,
                                                      size_t length,
                                                      double s_mean,
                                                      double s_std,
                                                      size_t j,
                                                      double mean,
                                                      double std,
                                                      double* X,
                                                      size_t timestep_stride,
                                                      double* X_buffer,
                                                      double min_dist) nogil:
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
        if dist >= min_dist:
            break

        x = (X[offset + timestep_stride * i] - s_mean) / s_std
        if not std_zero:
            x -= (X_buffer[i + j] - mean) / std
        dist += x * x

    return dist

cdef Shapelet shapelet_info_extract_unscaled_shapelet(
    ShapeletInfo s, const SlidingDistance t):
    cdef Shapelet shapelet = Shapelet(s.length)
    cdef double* data = shapelet.data
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)
    cdef size_t i
    cdef size_t p
    with nogil:
        for i in range(s.length):
            p = shapelet_offset + t.timestep_stride * i
            data[i] = t.X[p]

    return shapelet

cdef Shapelet shapelet_info_extract_shapelet(ShapeletInfo s,
                                             const SlidingDistance t):
    """Extract (i.e., allocate) a shapelet to be stored outside the
    store. The `ShapeletInfo` is extpected to have `mean` and `std`
    computed.

    :param s: information about a shapelet
    :param t: the time series storage
    :return: a normalized shapelet
    """
    cdef Shapelet shapelet = ScaledShapelet(s.length, s.mean, s.std)
    cdef double* data = shapelet.data
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)

    cdef size_t i
    cdef size_t p
    with nogil:
        if s.std == 0:
            for i in range(s.length):
                data[i] = 0.0
        else:
            for i in range(s.length):
                p = shapelet_offset + t.timestep_stride * i
                data[i] = (t.X[p] - s.mean) / s.std

    return shapelet


cdef int shapelet_info_update_statistics(ShapeletInfo* s,
                                         const SlidingDistance t) nogil:
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)
    cdef double ex = 0
    cdef double ex2 = 0
    cdef size_t i
    for i in range(s.length):
        current_value = t.X[shapelet_offset + i * t.timestep_stride]
        ex += current_value
        ex2 += current_value**2

    s[0].mean = ex / s.length
    s[0].std = sqrt(ex2 / s.length - s[0].mean * s[0].mean)
    return 0


cdef int shapelet_info_distances(ShapeletInfo s,
                                 const size_t* indicies,
                                 size_t n_indicies,
                                 const SlidingDistance t,
                                 double* result) nogil:
    cdef size_t p

    # TODO: consider prange
    for p in range(n_indicies):
        result[p] = shapelet_info_distance(s, t, indicies[p])
    return 0


cdef int shapelet_info_unscaled_distances(ShapeletInfo s,
                                          const size_t* samples,
                                          size_t n_samples,
                                          const SlidingDistance t,
                                          double* result) nogil:
    cdef size_t p
    for p in range(n_samples):
        result[p] = shapelet_info_unscaled_distance(s, t, samples[p])


cdef double shapelet_info_unscaled_distance(ShapeletInfo s,
                                            const SlidingDistance t,
                                            size_t t_index) nogil:
    cdef size_t sample_offset = t_index * t.sample_stride
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)

    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef size_t i
    cdef size_t j
    cdef double x
    for i in range(t.n_timestep - s.length + 1):
        dist = 0
        for j in range(s.length):
            if dist >= min_dist:
                 break

            x = t.X[sample_offset + t.timestep_stride * i + j]
            x -= t.X[shapelet_offset + t.timestep_stride * j]
            dist += x * x

        if dist < min_dist:
            min_dist = dist

    return sqrt(min_dist)


cdef double shapelet_info_distance(ShapeletInfo s,
                                   const SlidingDistance t,
                                   size_t t_index) nogil:
    cdef size_t sample_offset = t_index * t.sample_stride
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)

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

    for i in range(t.n_timestep):
        current_value = t.X[sample_offset + t.timestep_stride * i]
        ex += current_value
        ex2 += current_value * current_value

        buffer_pos = i % s.length
        t.X_buffer[buffer_pos] = current_value
        t.X_buffer[buffer_pos + s.length] = current_value
        if i >= s.length - 1:
            j = (i + 1) % s.length
            mean = ex / s.length
            std = sqrt(ex2 / s.length - mean * mean)
            dist = shapelet_info_subsequence_distance(
                shapelet_offset,
                s.length,
                s.mean,
                s.std,
                j,
                mean,
                std,
                t.X,
                t.timestep_stride,
                t.X_buffer,
                min_dist)

            if dist < min_dist:
                min_dist = dist

            current_value = t.X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return sqrt(min_dist)


cdef SlidingDistance new_sliding_distance( np.ndarray[np.float64_t,
                                                      ndim=2, mode="c"] X):
    cdef SlidingDistance sd
    sd.n_samples = <size_t> X.shape[0]
    sd.n_timestep = <size_t> X.shape[1]
    sd.X = <double*> X.data
    sd.sample_stride = <size_t> X.strides[0] / <size_t> X.itemsize
    sd.timestep_stride = <size_t> X.strides[1] / <size_t> X.itemsize
    sd.X_buffer = <double*> malloc(sizeof(double) * 2 * sd.n_timestep)

    if sd.X_buffer == NULL:
        raise MemoryError()
    return sd


cdef int free_sliding_distance(SlidingDistance sd) nogil:
    free(sd.X_buffer)
    sd.X_buffer = NULL
    # sd.X is freed by its owner
    return 0
