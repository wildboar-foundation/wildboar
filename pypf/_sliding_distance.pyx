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
from libc.math cimport INFINITY

from sklearn.utils import check_array


cpdef Shapelet make_scaled_shapelet_(size_t length, double mean,
                                     double std, object array):
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


cpdef Shapelet make_shapelet_(size_t length, object array):
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
        return (make_shapelet_, (self.length, self.array))

    @property
    def array(self):
        cdef np.ndarray[np.float64_t] arr = np.empty(self.length,
                                                     dtype=np.float64)
        cdef size_t i
        for i in range(self.length):
            arr[i] = self.data[i]
        return arr

    cdef double distance(self, const SlidingDistance t, size_t t_index) nogil:
        cdef size_t sample_offset = t_index * t.sample_stride
        # TODO: include `dim` and `dim_stride`
        return sliding_distance(
            0,
            1,
            self.length,
            self.data,
            sample_offset,
            t.timestep_stride,
            t.n_timestep,
            t.X,
            NULL)

    cdef void distances(self,
                       const SlidingDistance t,
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

    def __init__(self, size_t length, double mean, double std):
        """Initializes a shapelet with an empty c-array `data`.

        :param size_t length: the size of the shapelet
        """
        self.mean = mean
        self.std = std

    def __reduce__(self):
        return (make_scaled_shapelet_, (self.length, self.mean,
                                        self.std, self.array))

    cdef double distance(self, const SlidingDistance t, size_t t_index) nogil:
        cdef size_t sample_offset = t_index * t.sample_stride
        # TODO: include `dim_stride`
        return scaled_sliding_distance(
            0,
            1,
            self.length,
            self.mean,
            self.std,
            self.data,
            sample_offset,
            t.timestep_stride,
            t.n_timestep,
            t.X,
            t.X_buffer,
            NULL)


cdef Shapelet shapelet_info_extract_shapelet(ShapeletInfo s, const
                                             SlidingDistance t):
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


cdef Shapelet shapelet_info_extract_scaled_shapelet(ShapeletInfo s,
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
        for i in range(s.length):
            p = shapelet_offset + t.timestep_stride * i
            data[i] = t.X[p]
    return shapelet


cdef ShapeletInfo new_shapelet_info(size_t index, size_t start,
                                    size_t length) nogil:
    cdef ShapeletInfo s
    s.index = index
    s.start = start
    s.length = length
    s.mean = 0
    s.std = 0
    return s


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


cdef int shapelet_info_scaled_distances(ShapeletInfo s,
                                        const size_t* indicies,
                                        size_t n_indicies,
                                        const SlidingDistance t,
                                        double* result) nogil:
    cdef size_t p

    for p in range(n_indicies):
        result[p] = shapelet_info_scaled_distance(s, t, indicies[p])
    return 0


cdef int shapelet_info_distances(ShapeletInfo s,
                                          const size_t* samples,
                                          size_t n_samples,
                                          const SlidingDistance t,
                                          double* result) nogil:
    cdef size_t p
    for p in range(n_samples):
        result[p] = shapelet_info_distance(s, t, samples[p])


cdef double shapelet_info_distance(ShapeletInfo s,
                                   const SlidingDistance t,
                                   size_t t_index) nogil:
    cdef size_t sample_offset = t_index * t.sample_stride
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)
    return sliding_distance(
        shapelet_offset,
        t.timestep_stride,
        s.length,
        t.X,
        sample_offset,
        t.timestep_stride,
        t.n_timestep,
        t.X,
        NULL)


cdef double shapelet_info_scaled_distance(ShapeletInfo s,
                                   const SlidingDistance t,
                                   size_t t_index) nogil:
    cdef size_t sample_offset = t_index * t.sample_stride
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)

    # TODO: sample_offset inclued `dim`
    # TODO: shapelet_offset include `dim`
    return scaled_sliding_distance(
        shapelet_offset,
        t.timestep_stride,
        s.length,
        s.mean,
        s.std,
        t.X,
        sample_offset,
        t.timestep_stride,
        t.n_timestep,
        t.X,
        t.X_buffer,
        NULL)


# TODO: remove ndim=2 (no limit)
cdef SlidingDistance new_sliding_distance( np.ndarray[np.float64_t,
                                                      ndim=2, mode="c"] X):
    cdef SlidingDistance sd
    sd.n_samples = <size_t> X.shape[0]
    sd.n_timestep = <size_t> X.shape[1]
    sd.X = <double*> X.data
    sd.sample_stride = <size_t> X.strides[0] / <size_t> X.itemsize
    sd.timestep_stride = <size_t> X.strides[1] / <size_t> X.itemsize
    # TODO: compute `dim_stride`
    # TODO: set `n_dims`
    
    sd.X_buffer = <double*> malloc(sizeof(double) * 2 * sd.n_timestep)

    if sd.X_buffer == NULL:
        raise MemoryError()
    return sd


cdef int free_sliding_distance(SlidingDistance sd) nogil:
    free(sd.X_buffer)
    sd.X_buffer = NULL
    return 0


cdef double scaled_sliding_distance(size_t s_offset,
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
            dist = scaled_distance(s_offset, s_length, s_mean, s_std,
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


cdef inline double scaled_distance(size_t offset,
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


cdef double sliding_distance(size_t s_offset,
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


cdef int sliding_distance_matches(size_t s_offset,
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


cdef double scaled_sliding_distance_matches(size_t s_offset,
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
            dist = scaled_distance(s_offset, s_length, s_mean, s_std,
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
