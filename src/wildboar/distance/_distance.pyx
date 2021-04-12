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

import numpy as np

cimport numpy as np
from libc.math cimport NAN, sqrt
from libc.stdlib cimport free, malloc

from . import _dtw_distance, _euclidean_distance

from .._data cimport TSDatabase, ts_database_new
from ._distance cimport DistanceMeasure, TSCopy

from sklearn.utils import check_array

from .._utils import check_array_fast

_DISTANCE_MEASURE = {
    'euclidean': _euclidean_distance.EuclideanDistance,
    'dtw': _dtw_distance.DtwDistance,
    'scaled_euclidean': _euclidean_distance.ScaledEuclideanDistance,
    'scaled_dtw': _dtw_distance.ScaledDtwDistance,
}


cdef int ts_copy_init(
    TSCopy *ts_copy,
    Py_ssize_t dim,
    Py_ssize_t length,
    double mean,
    double std,
) nogil:
    ts_copy.dim = dim
    ts_copy.length = length
    ts_copy.mean = mean
    ts_copy.std = std
    ts_copy.data = <double*> malloc(sizeof(double) * length)
    ts_copy.extra = NULL
    if ts_copy.data == NULL:
        return -1


cdef void ts_copy_free(TSCopy *ts_copy) nogil:
    if ts_copy.data != NULL:
        free(ts_copy[0].data)
        ts_copy[0].data = NULL

    if ts_copy.extra != NULL:
        free(ts_copy[0].extra)
        ts_copy[0].extra = NULL


cdef void ts_view_init(TSView *ts_view) nogil:
    """Initialize  a shapelet info struct """
    ts_view.start = 0
    ts_view.length = 0
    ts_view.dim = 0
    ts_view.mean = NAN
    ts_view.std = NAN
    ts_view.index = 0
    ts_view.extra = NULL


cdef void ts_view_free(TSView *ts_view) nogil:
    """Free the `extra` payload of a shapelet info if needed """
    if ts_view.extra != NULL:
        free(ts_view.extra)
        ts_view.extra = NULL


cdef int _ts_view_update_statistics(TSView *ts_view, const TSDatabase *td) nogil:
    """Update the mean and standard deviation of a shapelet info struct """
    cdef Py_ssize_t shapelet_offset = (
            ts_view.index * td.sample_stride +
            ts_view.dim * td.dim_stride +
            ts_view.start * td.timestep_stride
    )
    cdef double ex = 0
    cdef double ex2 = 0
    cdef Py_ssize_t i
    for i in range(ts_view.length):
        current_value = td.data[shapelet_offset + i * td.timestep_stride]
        ex += current_value
        ex2 += current_value ** 2

    ts_view.mean = ex / ts_view.length
    ex2 = ex2 / ts_view.length - ts_view.mean * ts_view.mean
    if ex2 > 0:
        ts_view.std = sqrt(ex2)
    else:
        ts_view.std = 1.0
    return 0


cdef class DistanceMeasure:
    """A distance measure can compute the distance between time series and
    shapelets """

    def __cinit__(self, Py_ssize_t n_timestep, *args, **kvargs):
        self.n_timestep = n_timestep


    def __reduce__(self):
        return self.__class__, (self.n_timestep,)


    cdef int init(self, TSDatabase *td) nogil:
        return 0


    cdef void ts_view_sub_distances(
        self,
        TSView *ts_view,
        TSDatabase *td,
        Py_ssize_t *samples,
        double *distances,
        Py_ssize_t n_samples,
    ) nogil:
        """ Compute the distance between the shapelet all samples in `samples`

        :param ts_view: information about the shapelet
        :param td: the time series database
        :param samples: array of length `n_samples` samples to compute
        the distance to
        :param distances: array to store the distances. The the length
        of distances >= `n_samples`, the `i = 0,...,n_samples`
        position stores the distance between the i:th sample (in
        `samples`) and `s` [out param]
        :param n_samples: the number of samples 
        """
        cdef Py_ssize_t p
        for p in range(n_samples):
            distances[p] = self.ts_view_sub_distance(ts_view, td, samples[p])


    cdef int init_ts_view(
        self,
        TSDatabase *_td,
        TSView *ts_view,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil:
        """Set the parameters of the view

        :param _td: shapelet database
        :param ts_view: [out param] 
        :param index: the index of the sample in `td`
        :param start: the start position of the subsequence
        :param length: the length of the subsequence
        :param dim: the dimension of the subsequence
        :return non-negative on success
        """
        ts_view.index = index
        ts_view.dim = dim
        ts_view.start = start
        ts_view.length = length
        ts_view.mean = NAN
        ts_view.std = NAN
        ts_view.extra = NULL
        return 0


    cdef int init_ts_copy_from_obj(
        self,
        TSCopy *ts_copy,
        object obj,
    ):
        dim, arr = obj
        ts_copy.dim = dim
        ts_copy.length = arr.shape[0]
        ts_copy.mean = NAN
        ts_copy.std = NAN
        ts_copy.data = <double*> malloc(ts_copy.length * sizeof(double))
        ts_copy.extra = NULL
        if ts_copy.data == NULL:
            return -1

        cdef Py_ssize_t i
        for i in range(ts_copy[0].length):
            ts_copy[0].data[i] = arr[i]
        return 0


    cdef object object_from_ts_copy(
        self, 
        TSCopy *ts_copy
    ):
        cdef Py_ssize_t j
        arr = np.empty(ts_copy[0].length, dtype=np.float64)
        for j in range(ts_copy[0].length):
            arr[j] = ts_copy[0].data[j]

        return (ts_copy.dim, arr)


    cdef int init_ts_copy(
        self,
        TSCopy *ts_copy,
        TSView *ts_view,
        TSDatabase *td,
    ) nogil:
        ts_copy_init(ts_copy, ts_view.dim, ts_view.length, ts_view.mean, ts_view.std)
        ts_copy.ts_start = ts_view.start
        ts_copy.ts_index = ts_view.index
        cdef double *data = ts_copy.data
        cdef Py_ssize_t tc_offset = (
            ts_view.index * td.sample_stride +
            ts_view.start * td.timestep_stride +
            ts_view.dim * td.dim_stride
        )

        cdef Py_ssize_t i
        cdef Py_ssize_t p

        for i in range(ts_view.length):
            p = tc_offset + td.timestep_stride * i
            data[i] = td.data[p]

        return 0


    cdef double ts_view_sub_distance(
        self,
        TSView *ts_view,
        TSDatabase *td,
        Py_ssize_t t_index,
    ) nogil:
        """Return the minimum distance
        
        Parameters
        ----------
        ts_view : the subsequence
         
        td : time series database
        
        t_index : sample to compare

        Returns
        -------
        distance : the distance
        """
        with gil:
            raise NotImplementedError()


    cdef double ts_copy_sub_distance(
        self,
        TSCopy *ts_view,
        TSDatabase *td,
        Py_ssize_t t_index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        """Return the distance between `s` and the sample specified by
        `t_index` in `td` setting the index of the best matching
        position to `return_index` unless `return_index == NULL`

        :param ts_view: the shapelet

        :param td: the time series database

        :param t_index: the sample index

        :param return_index: (out) the index of the best matching position
        """
        with gil:
            raise NotImplementedError()

    cdef int ts_copy_sub_matches(
        self,
        TSCopy *s_ptr,
        TSDatabase *td_ptr,
        Py_ssize_t t_index,
        double threshold,
        Py_ssize_t** matches,
        double** distances,
        Py_ssize_t *n_matches,
    ) nogil except -1:
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


    cdef double ts_copy_distance(self, TSCopy *s, TSDatabase *td, Py_ssize_t t_index) nogil:
        return self.ts_copy_sub_distance(s, td, t_index)


    cdef bint support_unaligned(self) nogil:
        return 0


cdef class ScaledDistanceMeasure(DistanceMeasure):
    """Distance measure that uses computes the distance on mean and
    variance standardized shapelets"""

    cdef int init_ts_copy_from_obj(
        self,
        TSCopy *ts_copy,
        object obj,
    ):
        cdef int err = DistanceMeasure.init_ts_copy_from_obj(
            self, ts_copy, obj
        )
        if err == -1:
            return -1
        dim, arr = obj
        ts_copy.mean = np.mean(arr)
        ts_copy.std = np.std(arr)
        if ts_copy.std == 0.0:
            ts_copy.std = 1.0
        return 0

    cdef int init_ts_view(
        self,
        TSDatabase *td,
        TSView *ts_view,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil:
        DistanceMeasure.init_ts_view(self, td, ts_view, index, start, length, dim)
        _ts_view_update_statistics(ts_view, td)
        return -1


cdef class FuncDistanceMeasure(DistanceMeasure):
    cdef object func
    cdef np.ndarray x_buffer
    cdef np.ndarray y_buffer
    cdef bint _support_unaligned

    def __cinit__(self, Py_ssize_t n_timestep, object func, bint support_unaligned=False):
        self.n_timestep = n_timestep
        self.func = func
        self.x_buffer = np.empty(n_timestep, dtype=np.float64)
        self.y_buffer = np.empty(n_timestep, dtype=np.float64)
        self._support_unaligned = support_unaligned


    def __reduce__(self):
        return self.__class__, (self.n_timestep, self.func)


    cdef double ts_copy_sub_distance(
        self,
        TSCopy *ts_copy,
        TSDatabase *td,
        Py_ssize_t t_index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        cdef Py_ssize_t i
        cdef Py_ssize_t sample_offset = (t_index * td.sample_stride + ts_copy.dim * td.dim_stride)
        with gil:
            for i in range(td.n_timestep):
                if i < ts_copy.length:
                    self.x_buffer[i] = ts_copy.data[i]
                self.y_buffer[i] = td.data[sample_offset + td.timestep_stride * i]

            return self.func(self.x_buffer[:ts_copy.length], self.y_buffer)

    cdef double ts_view_sub_distance(
        self,
        TSView *ts_view,
        TSDatabase *td,
        Py_ssize_t t_index,
    ) nogil:
        cdef Py_ssize_t i
        cdef Py_ssize_t sample_offset = (t_index * td.sample_stride +
                                         ts_view.dim * td.dim_stride)
        cdef Py_ssize_t shapelet_offset = (ts_view.index * td.sample_stride +
                                           ts_view.dim * td.dim_stride +
                                           ts_view.start * td.timestep_stride)
        with gil:
            for i in range(td.n_timestep):
                if i < ts_view.length:
                    self.x_buffer[i] = td.data[shapelet_offset + td.timestep_stride * i]
                self.y_buffer[i] = td.data[sample_offset + td.timestep_stride * i]

            return self.func(self.x_buffer[:ts_view.length], self.y_buffer)

    cdef double ts_copy_distance(
        self,
        TSCopy *ts_copy,
        TSDatabase *td,
        Py_ssize_t t_index,
    ) nogil:
        return self.ts_copy_sub_distance(ts_copy, td, t_index)

    cdef bint support_unaligned(self) nogil:
        return self._support_unaligned


def _validate_shapelet(shapelet):
    cdef np.ndarray s = check_array(
        shapelet, ensure_2d=False, dtype=np.float64, order="c")
    if s.ndim > 1:
        raise ValueError("only 1d shapelets allowed")

    if not s.flags.c_contiguous:
        s = np.ascontiguousarray(s, dtype=np.float64)
    return s


def _validate_data(data):
    cdef np.ndarray x = check_array(
        data, ensure_2d=False, allow_nd=True, dtype=np.float64, order="c")
    if x.ndim == 1:
        x = x.reshape(-1, x.shape[0])

    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x, dtype=np.float64)
    return x


def _check_sample(sample, n_samples):
    if sample < 0 or sample >= n_samples:
        raise ValueError("illegal sample {}".format(sample))


def _check_dim(dim, ndims):
    if dim < 0 or dim >= ndims:
        raise ValueError("illegal dimension {}".format(dim))


cdef np.ndarray _new_match_array(Py_ssize_t *matches, Py_ssize_t n_matches):
    if n_matches > 0:
        match_array = np.empty(n_matches, dtype=np.intp)
        for i in range(n_matches):
            match_array[i] = matches[i]
        return match_array
    else:
        return None


cdef np.ndarray _new_distance_array(
        double *distances, Py_ssize_t n_matches):
    if n_matches > 0:
        dist_array = np.empty(n_matches, dtype=np.float64)
        for i in range(n_matches):
            dist_array[i] = distances[i]
        return dist_array
    else:
        return None


cpdef DistanceMeasure get_distance_measure(
    Py_ssize_t n_timestep,
    object metric,
    dict metric_params=None,
):
    """Create a new distance measure

    Parameters
    ----------
    metric : str or callable
        A metric name or callable

    td : TSDatabase
        Number of maximum number of timesteps in the database

    metric_params : dict, optional
        Parameters to the metric

    Returns
    -------
    distance_measure : a distance measure instance
    """
    cdef DistanceMeasure distance_measure
    metric_params = metric_params or {}
    if isinstance(metric, str):
        if metric in _DISTANCE_MEASURE:
            distance_measure = _DISTANCE_MEASURE[metric](n_timestep, **metric_params)
        else:
            raise ValueError("metric (%s) is not supported" % metric)
    elif hasattr(metric, "__call__"):
        distance_measure = FuncDistanceMeasure(n_timestep, metric)
    elif isinstance(metric, DistanceMeasure):
        return metric # TODO: check n_timestep
    else:
        raise ValueError("unknown metric, got %r" % metric)
    return distance_measure


def distance(shapelet, data, dim=0, sample=None, metric="euclidean", metric_params=None, subsequence_distance=True, return_index=False):
    cdef np.ndarray s = _validate_shapelet(shapelet)
    cdef np.ndarray x = _validate_data(data)
    if sample is None:
        if x.shape[0] == 1:
            sample = 0
        else:
            sample = np.arange(x.shape[0])
    cdef TSDatabase sd = ts_database_new(x)

    _check_dim(dim, sd.n_dims)
    cdef double min_dist
    cdef Py_ssize_t min_index

    cdef double mean = 0
    cdef double std = 0

    cdef DistanceMeasure distance_measure = get_distance_measure(
        sd.n_timestep, metric, metric_params
    )

    if (
            not subsequence_distance and
            not distance_measure.support_unaligned() and
            s.shape[0] != sd.n_timestep
    ):
        raise ValueError(
            "x.shape[0] != y.shape[-1], got %r, %r" % (s.shape[0], sd.n_timestep)
        )

    if not distance_measure.support_unaligned() and s.shape[0] > sd.n_timestep:
        raise ValueError(
            "x.shape[0] > y.shape[-1], got %r, %r" % (s.shape[0], sd.n_timestep)
        )


    cdef TSCopy shape # TODO: free me
    distance_measure.init_ts_copy_from_obj(&shape, (dim, s))
    if isinstance(sample, int):
        if subsequence_distance:
            min_dist = distance_measure.ts_copy_sub_distance(
                &shape, &sd, sample, &min_index)
        else:
            min_dist = distance_measure.ts_copy_distance(
                &shape, &sd, sample
            )
            min_index = 0

        if return_index:
            return min_dist, min_index
        else:
            return min_dist
    else:  # assume an `array_like` object for `samples`
        samples = check_array(sample, ensure_2d=False, dtype=int)
        dist = []
        ind = []
        # TODO: add n_jobs
        for i in samples:
            if subsequence_distance:
                min_dist = distance_measure.ts_copy_sub_distance(
                    &shape, &sd, i, &min_index
                )
            else:
                min_dist = distance_measure.ts_copy_distance(
                    &shape, &sd, i
                )
                min_index = 0

            dist.append(min_dist)
            ind.append(min_index)

        if return_index:
            return np.array(dist), np.array(ind)
        else:
            return np.array(dist)


def matches(shapelet, X, threshold, dim=0, sample=None, metric="euclidean", metric_params=None, return_distance=False):
    cdef np.ndarray s = _validate_shapelet(shapelet)
    cdef np.ndarray x = _validate_data(X)
    _check_dim(dim, x.ndim)
    if sample is None:
        if x.shape[0] == 1:
            sample = 0
        else:
            sample = np.arange(x.shape[0])

    cdef TSDatabase sd = ts_database_new(x)

    cdef Py_ssize_t *matches
    cdef double *distances
    cdef Py_ssize_t n_matches

    cdef DistanceMeasure distance_measure = get_distance_measure(
        sd.n_timestep, metric, metric_params
    )
    cdef TSCopy shape
    distance_measure.init_ts_copy_from_obj(&shape, (dim, s))
    cdef Py_ssize_t i
    if isinstance(sample, int):
        _check_sample(sample, sd.n_samples)
        distance_measure.ts_copy_sub_matches(
            &shape, &sd, sample, threshold, &matches, &distances, &n_matches)

        match_array = _new_match_array(matches, n_matches)
        distance_array = _new_distance_array(distances, n_matches)
        free(distances)
        free(matches)

        if return_distance:
            return distance_array, match_array
        else:
            return match_array
    else:
        samples = check_array(sample, ensure_2d=False, dtype=int)
        match_list = []
        distance_list = []
        for i in samples:
            _check_sample(i, sd.n_samples)
            distance_measure.ts_copy_sub_matches(&shape, &sd, i, threshold, &matches, &distances, &n_matches)
            match_array = _new_match_array(matches, n_matches)
            distance_array = _new_distance_array(distances, n_matches)
            match_list.append(match_array)
            distance_list.append(distance_array)
            free(matches)
            free(distances)

        if return_distance:
            return distance_list, match_list
        else:
            return match_list
