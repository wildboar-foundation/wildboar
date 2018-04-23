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

from libc.stdlib cimport realloc
from libc.stdlib cimport malloc
from libc.stdlib cimport free

from libc.math cimport sqrt
from libc.math cimport INFINITY, NAN


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


cpdef Shapelet make_scaled_shapelet_(object me, size_t dim, size_t length, double mean,
                                     double std, object array):
    """Reconstruct a `Shapelet`-object from Pickle

    :param length: the size of the shapelet
    :param array: the Numpy array
    :return: a shapelet
    """
    cdef Shapelet shapelet = me(dim, length, mean, std)
    cdef size_t i
    for i in range(<size_t> array.shape[0]):
        shapelet.data[i] = array[i]

    return shapelet


cpdef Shapelet make_shapelet_(object me, size_t dim, size_t length, object array):
    """Reconstruct a `Shapelet`-object from Pickle

    :param length: the size of the shapelet
    :param array: the Numpy array
    :return: a shapelet
    """
    cdef Shapelet shapelet = me(dim, length)
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
        return make_shapelet_, (self.__class__, self.dim, self.length,
                                self.array)

    @property
    def array(self):
        cdef np.ndarray[np.float64_t] arr = np.empty(
            self.length, dtype=np.float64)
        cdef size_t i
        for i in range(self.length):
            arr[i] = self.data[i]
        return arr

    cdef double distance(self, const TSDatabase t, size_t t_index) nogil:
        pass

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
        return make_scaled_shapelet_, (self.__class__, self.dim,
                                       self.length, self.mean,
                                       self.std, self.array)


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