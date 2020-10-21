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
# wildboar is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc
from libc.stdlib cimport free

from libc.math cimport sqrt
from libc.math cimport NAN

cdef int shapelet_init(Shapelet *shapelet, size_t dim, size_t length, double mean, double std) nogil:
    shapelet[0].dim = dim
    shapelet[0].length = length
    shapelet[0].mean = mean
    shapelet[0].std = std
    shapelet[0].data = <double*> malloc(sizeof(double) * length)
    if shapelet[0].data == NULL:
        return -1

cdef void shapelet_free(Shapelet *shapelet) nogil:
    if shapelet != NULL and shapelet[0].data != NULL:
        free(shapelet[0].data)

cdef void shapelet_info_init(ShapeletInfo *s) nogil:
    """Initialize  a shapelet info struct """
    s[0].start = 0
    s[0].length = 0
    s[0].dim = 0
    s[0].mean = NAN
    s[0].std = NAN
    s[0].index = 0

cdef void shapelet_info_free(ShapeletInfo *shapelet_info) nogil:
    """Free the `extra` payload of a shapelet info if needed """
    if shapelet_info[0].extra != NULL:
        free(shapelet_info[0].extra)
        shapelet_info[0].extra = NULL

cdef int _shapelet_info_update_statistics(ShapeletInfo *s, const TSDatabase *t_ptr) nogil:
    """Update the mean and standard deviation of a shapelet info struct """
    cdef TSDatabase t = t_ptr[0]
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.dim * t.dim_stride +
                                   s.start * t.timestep_stride)
    cdef double ex = 0
    cdef double ex2 = 0
    cdef size_t i
    for i in range(s.length):
        current_value = t.data[shapelet_offset + i * t.timestep_stride]
        ex += current_value
        ex2 += current_value ** 2

    s[0].mean = ex / s.length
    s[0].std = sqrt(ex2 / s.length - s[0].mean * s[0].mean)
    return 0

cdef TSDatabase ts_database_new(np.ndarray data):
    """Construct a new time series database from a ndarray """
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

cdef class DistanceMeasure:
    """A distance measure can compute the distance between time series and
    shapelets """

    def __cinit__(self, size_t n_timestep, *args, **kvargs):
        """ Initialize a new distance measure

        :param n_timesteps: the (maximum) number of timepoints in a timeseries

        :param *args: optimal arguments for subclasses

        :param **kvargs: optional arguments for subclasses
        """
        self.n_timestep = n_timestep

    def __reduce__(self):
        return self.__class__, (self.n_timestep, )

    cdef void shapelet_info_distances(
            self, ShapeletInfo *s, TSDatabase *td, size_t *samples,
            double *distances, size_t n_samples) nogil:
        """ Compute the distance between the shapelet `s` in `td` and all
        samples in `samples`

        :param s: information about the shapelet
        :param td: the time series database
        :param samples: array of length `n_samples` samples to compute
        the distance to
        :param distances: array to store the distances. The the length
        of distances >= `n_samples`, the `i = 0,...,n_samples`
        position stores the distance between the i:th sample (in
        `samples`) and `s` [out param]
        :param n_samples: the number of samples 
        """
        cdef size_t p
        for p in range(n_samples):
            distances[p] = self.shapelet_info_distance(s, td, samples[p])

    cdef int init_shapelet_info(
            self, TSDatabase *_td, ShapeletInfo *shapelet_info, size_t index, size_t start,
            size_t length, size_t dim) nogil:
        """Return a information about a shapelet

        :param _td: shapelet database
        :param shapelet_info: [out param] 
        :param index: the index of the sample in `td`
        :param start: the start position of the subsequence
        :param length: the length of the subsequence
        :param dim: the dimension of the subsequence
        :return non-negative on success
        """
        shapelet_info[0].index = index
        shapelet_info[0].dim = dim
        shapelet_info[0].start = start
        shapelet_info[0].length = length
        shapelet_info[0].mean = NAN
        shapelet_info[0].std = NAN
        shapelet_info[0].extra = NULL
        return 0

    cdef int init_shapelet_ndarray(self, Shapelet *shapelet, np.ndarray arr, size_t dim):
        shapelet[0].dim = dim
        shapelet[0].length = arr.shape[0]
        shapelet[0].mean = NAN
        shapelet[0].std = NAN
        shapelet[0].data = <double*> malloc(shapelet[0].length * sizeof(double))
        if shapelet[0].data == NULL:
            return -1

        cdef size_t i
        for i in range(shapelet[0].length):
            shapelet[0].data[i] = arr[i]
        return 0

    cdef int init_shapelet(self, Shapelet *shapelet, ShapeletInfo *s_ptr, TSDatabase *td_ptr) nogil:
        cdef ShapeletInfo s = s_ptr[0]
        cdef TSDatabase td = td_ptr[0]
        shapelet_init(shapelet, s.dim, s.length, s.mean, s.std)
        shapelet[0].ts_start = s.start
        shapelet[0].ts_index = s.index
        cdef double *data = shapelet[0].data
        cdef size_t shapelet_offset = (s.index * td.sample_stride +
                                       s.start * td.timestep_stride +
                                       s.dim * td.dim_stride)

        cdef size_t i
        cdef size_t p

        for i in range(s.length):
            p = shapelet_offset + td.timestep_stride * i
            data[i] = td.data[p]

        return 0

    cdef double shapelet_info_distance(
            self, ShapeletInfo *si, TSDatabase *td_ptr, size_t t_index) nogil:
        """Return the distance between `s` and the sample specified by the
        index `t_index` in `td`. Implemented by subclasses.

        :param si: shapelet information

        :param td_ptr: the time series database

        :param t_index: the index of the time series
        """
        with gil:
            raise NotImplementedError()

    cdef double shapelet_distance(
            self, Shapelet *s_ptr, TSDatabase *td_ptr, size_t t_index,
            size_t *return_index=NULL) nogil:
        """Return the distance between `s` and the sample specified by
        `t_index` in `td` setting the index of the best matching
        position to `return_index` unless `return_index == NULL`

        :param s_ptr: the shapelet

        :param td_ptr: the time series database

        :param t_index: the sample index

        :param return_index: (out) the index of the best matching position
        """
        with gil:
            raise NotImplementedError()

    cdef int shapelet_matches(
            self, Shapelet *s_ptr, TSDatabase *td_ptr, size_t t_index,
            double threshold, size_t** matches, double** distances,
            size_t *n_matches) nogil except -1:
        """Compute the matches for `s` in the sample `t_index` in `td` where
        the distance threshold is below `threshold`, storing the
        matching starting positions in `matches` and distance (<
        `threshold`) in `distances` with `n_matches` storing the
        number of successful matches.

        Note:

        - `matches` will be allocated and must be freed by the caller
        - `distances` will be allocated and must be freed by the caller

        :param s_ptr: the shapelet

        :param td_ptr: the time series database

        :param t_index: the sample

        :param threshold: the minimum distance to consider a match

        :param matches: (out) array of matching locations

        :param distances: (out) array of distance at the matching
        location (< `threshold`)

        :param n_matches: (out) the number of matches
        """
        with gil:
            raise NotImplementedError()

cdef class ScaledDistanceMeasure(DistanceMeasure):
    """Distance measure that uses computes the distance on mean and
    variance standardized shapelets"""

    cdef int init_shapelet_ndarray(self, Shapelet *shapelet, np.ndarray arr, size_t dim):
        cdef int err = DistanceMeasure.init_shapelet_ndarray(self, shapelet, arr, dim)
        if err == -1:
            return -1
        shapelet[0].mean = np.mean(arr)
        shapelet[0].std = np.std(arr)
        return 0

    cdef int init_shapelet_info(self, TSDatabase *td, ShapeletInfo *shapelet_info, size_t index, size_t start,
                                size_t length, size_t dim) nogil:
        DistanceMeasure.init_shapelet_info(self, td, shapelet_info, index, start, length, dim)
        _shapelet_info_update_statistics(shapelet_info, td)
        return -1

# cdef class FuncDistanceMeasure(DistanceMeasure):
#     """ Wrapper for python function """
#     cdef object func
#     cdef np.ndarray[ndim=1, dtype=np.float] sbuffer
#     cdef np.ndarray[ndim=1, dtype=np.float] tbuffer

#     def __cinit__(self, size_t n_timesteps, object func):
#         self.func = func
#         self.sbuffer = np.empty([n_timesteps], dtype=np.float32)
#         self.tbuffer = np.empty([n_timesteps], dtype=np.float32)

#     cdef double shapelet_info_distance(
#             self, ShapeletInfo s, TSDatabase td, size_t t_index) nogil:
#         with gil:
#             raise NotImplementedError()

#     cdef double shapelet_distance(
#             self, Shapelet s, TSDatabase td, size_t t_index,
#             size_t* return_index=NULL) nogil:
#         with gil:
#             raise NotImplementedError()

#     cdef int shapelet_matches(
#             self, Shapelet s, TSDatabase td, size_t t_index,
#             double threhold, size_t** matches,  double** distances,
#             size_t* n_matches) nogil except -1:
#         with gil:
#             raise NotImplementedError()    

# cdef class Shapelet:
#     def __cinit__(self,
#                   size_t dim,
#                   size_t length,
#                   double mean,
#                   double std,
#                   int ts_index=-1,
#                   int ts_start=-1):
#         self.length = length
#         self.mean = mean
#         self.std = std
#         self.dim = dim
#         self.ts_index = ts_index
#         self.ts_start = ts_start
#         self.data = <double*> malloc(sizeof(double) * length)
#         if self.data == NULL:
#             raise MemoryError()
#         self.extra = NULL
#
#     def __dealloc__(self):
#         free(self.data)
#         if self.extra != NULL:
#             free(self.extra)
#
#     def __reduce__(self):
#         return make_shapelet_, (self.__class__, self.dim, self.length,
#                                 self.mean, self.std, self.array,
#                                 self.ts_index, self.ts_start)
#
#     @property
#     def ts_info(self):
#         """Returns information about the time series from which the shapelet was extracted
#         :returns `None` or `(ts_index, ts_start, shapelet_length)`
#         """
#         if self.ts_index < 0 or self.ts_start < 0:
#             return None
#         else:
#             return self.ts_index, self.ts_start, self.length
#
#     @property
#     def array(self):
#         cdef np.ndarray[np.float64_t] arr = np.empty(
#             self.length, dtype=np.float64)
#         cdef size_t i
#         for i in range(self.length):
#             arr[i] = self.data[i]
#         return arr
