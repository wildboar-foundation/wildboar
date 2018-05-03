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

from pypf._utils cimport rand_int

cdef class DistanceMeasure:

    def __cinit__(self, size_t n_timesteps, *args, **kvargs):
        pass

    cdef void shapelet_info_distances(
        self, ShapeletInfo s, TSDatabase td, size_t* samples,
        double* distances, size_t n_samples) nogil:

        cdef size_t p
        for p in range(n_samples):
            distances[p] = self.shapelet_info_distance(s, td, samples[p])

    cdef ShapeletInfo new_shapelet_info(
        self, TSDatabase _td, size_t index, size_t start,
        size_t length, size_t dim) nogil:

        cdef ShapeletInfo shapelet_info
        shapelet_info.index = index
        shapelet_info.dim = dim
        shapelet_info.start = start
        shapelet_info.length = length
        shapelet_info.mean = NAN
        shapelet_info.std = NAN
        shapelet_info.extra = NULL
        return shapelet_info

    cdef Shapelet new_shapelet(self, ShapeletInfo s, TSDatabase td):
        cdef Shapelet shapelet = Shapelet(s.dim, s.length, NAN, NAN)
        cdef double*data = shapelet.data
        cdef size_t shapelet_offset = (s.index * td.sample_stride +
                                       s.start * td.timestep_stride +
                                       s.dim * td.dim_stride)
        cdef size_t i
        cdef size_t p
        with nogil:
            for i in range(s.length):
                p = shapelet_offset + td.timestep_stride * i
                data[i] = td.data[p]

        return shapelet

    cdef void _validate_input(self, np.ndarray t, TSDatabase td, size_t t_index, size_t dim):
        if t.ndim != 1 and t.shape[0] != 1:
            raise ValueError("invalid input array")

        if td.n_dims >= dim:
            raise ValueError("illegal dim")

        if td.n_timestep != len(t):
            raise ValueError("illegal input array size")

        if t_index >= td.n_samples:
            raise ValueError("illegal sample")

    cdef double distance(
            self, np.ndarray t, TSDatabase td, size_t t_index,
            size_t dim, size_t* return_index):
        raise NotImplemented()
    
    cdef double shapelet_info_distance(
            self, ShapeletInfo s, TSDatabase td, size_t t_index) nogil:
        with gil:
            raise NotImplemented()

    cdef double shapelet_distance(
            self, Shapelet s, TSDatabase td, size_t t_index) nogil:
        with gil:
            raise NotImplemented()


cdef class ScaledDistanceMeasure(DistanceMeasure):

    cdef Shapelet new_shapelet(self, ShapeletInfo s, TSDatabase td):
        cdef Shapelet shapelet = Shapelet(s.dim, s.length, s.mean, s.std)
        cdef double*data = shapelet.data
        cdef size_t shapelet_offset = (s.index * td.sample_stride +
                                       s.start * td.timestep_stride +
                                       s.dim * td.dim_stride)

        cdef size_t i
        cdef size_t p
        with nogil:
            for i in range(s.length):
                p = shapelet_offset + td.timestep_stride * i
                data[i] = td.data[p]

        return shapelet

    cdef ShapeletInfo new_shapelet_info(self,
                                        TSDatabase td,
                                        size_t index,
                                        size_t start,
                                        size_t length,
                                        size_t dim) nogil:
    
        cdef ShapeletInfo shapelet_info
        shapelet_info.index = index
        shapelet_info.dim = dim
        shapelet_info.start = start
        shapelet_info.length = length
        shapelet_info.extra = NULL

        shapelet_info_update_statistics(&shapelet_info, td)
        return shapelet_info


cpdef Shapelet make_shapelet_(object me,
                              size_t dim,
                              size_t length,
                              double mean,
                              double std,
                              object array):
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


cdef class Shapelet:

    def __cinit__(self, size_t dim, size_t length, double mean, double std):
        self.length = length
        self.mean = mean
        self.std = std
        self.dim = dim
        self.data = <double*> malloc(sizeof(double) * length)
        if self.data == NULL:
            raise MemoryError()

    def __dealloc__(self):
        free(self.data)

    def __reduce__(self):
        return make_shapelet_, (self.__class__, self.dim, self.length,
                                self.mean, self.std, self.array)

    @property
    def array(self):
        cdef np.ndarray[np.float64_t] arr = np.empty(
            self.length, dtype=np.float64)
        cdef size_t i
        for i in range(self.length):
            arr[i] = self.data[i]
        return arr


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


cdef TSDatabase ts_database_new(np.ndarray data):
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError("ndim {0} < 2 or {0} > 3".format(data.ndim))

    cdef TSDatabase sd
    sd.n_samples = <size_t> data.shape[0]
    sd.n_timestep = <size_t> data.shape[data.ndim - 1]
    sd.data = <double*> data.data
    sd.sample_stride = <size_t> data.strides[0] / <size_t> data.itemsize
    sd.timestep_stride = (<size_t> data.strides[data.ndim - 1] /
                          <size_t> data.itemsize)

    if data.ndim == 3:
        sd.n_dims = <size_t> data.shape[data.ndim - 2]
        sd.dim_stride = (<size_t> data.strides[data.ndim - 2] /
                         <size_t> data.itemsize)
    else:
        sd.n_dims = 1
        sd.dim_stride = 0

    return sd

cdef void shapelet_info_free(ShapeletInfo shapelet_info) nogil:
    if shapelet_info.extra != NULL:
        free(shapelet_info.extra)
        shapelet_info.extra = NULL
