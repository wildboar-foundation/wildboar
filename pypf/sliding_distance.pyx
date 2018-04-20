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

from libc.stdlib cimport free

from pypf._sliding_distance cimport TSDatabase
from pypf._sliding_distance cimport new_ts_database
from pypf._sliding_distance cimport free_ts_database
from pypf._sliding_distance cimport euclidean_distance
from pypf._sliding_distance cimport euclidean_distance_matches

from pypf._sliding_distance cimport scaled_euclidean_distance
from pypf._sliding_distance cimport scaled_euclidean_distance_matches

from sklearn.utils import check_array


# validate and convert shapelet to sutable format
def validate_shapelet_(shapelet):
    cdef np.ndarray s = check_array(
        shapelet, ensure_2d=False, dtype=np.float64, order="c")
    if s.ndim > 1:
        raise ValueError("only 1d shapelets allowed")

    if not s.flags.contiguous:
        s = np.ascontiguousarray(s, dtype=np.float64)
    return s


# validate and convert time series data to suitable
def validate_data_(data):
    cdef np.ndarray x = check_array(
        data, ensure_2d=False, allow_nd=True, dtype=np.float64, order="c")
    if x.ndim == 1:
        x = x.reshape(-1, x.shape[0])

    if not x.flags.contiguous:
        x = np.ascontiguousarray(x, dtype=np.float64)
    return x

def check_sample_(sample, n_samples):
    if sample < 0 or sample >= n_samples:
        raise ValueError("illegal sample {}".format(sample))

def check_dim(dim, ndims):
    if dim < 0 or dim >= ndims:
        raise ValueError("illegal dimension {}".format(dim))

cdef np.ndarray make_numpy_array_(size_t* matches,
                                  size_t n_matches):
    if n_matches > 0:
        match_array = np.empty(n_matches, dtype=np.intp)
        for i in range(n_matches):
            match_array[i] = matches[i]
        return match_array
    else:
        return np.empty([0], dtype=np.intp)


def min_distance(
        shapelet,
        data,
        dim=0,
        sample=None,
        scale=False,
        return_index=False):
    """Computes the minimum distance between `s` and the samples in `x`

    :param shapelet: the subsequence `array_like`
    :param data: the samples [n_samples, n_timesteps]
    :param dim: the time series dimension to search (default: 0)
    :param sample: the samples to compare to `int` or `array_like` or `None`.
                   If `None` compare to all. (default: `None`)
    :param scale: search in a scaled space
    :param return_index: if `true` return the index of the best
                         match. If there are many equally good
                         best matches, the first is returned.
    :returns: `float`,
              `(float, int)`,
              `float [n_samples]` or
              `(float [n_samples], int [n_samples]` depending on input

    """
    cdef np.ndarray s = validate_shapelet_(shapelet)
    cdef np.ndarray x = validate_data_(data)
    if sample is None:
        if x.shape[0] == 1:
            sample = 0
        else:
            sample = np.arange(x.shape[0])


    cdef TSDatabase sd = new_ts_database(x)

    check_dim(dim, sd.n_dims)
    cdef double min_dist
    cdef size_t min_index

    cdef size_t s_offset = 0
    cdef size_t s_stride = <size_t> s.strides[0] // s.itemsize
    cdef size_t s_length = s.shape[0]
    cdef double* s_data = <double*> s.data

    cdef size_t t_offset

    cdef double mean = 0
    cdef double std = 0

    if scale:
        mean = np.mean(s)
        std = np.std(s)

    try:
        if isinstance(sample, int):
            check_sample_(sample, sd.n_samples)

            t_offset = sample * sd.sample_stride + \
                       dim * sd.dim_stride
            if scale:
                min_dist = scaled_euclidean_distance(
                    s_offset,
                    s_stride,
                    s_length,
                    mean,
                    std,
                    s_data,
                    t_offset,
                    sd.timestep_stride,
                    sd.n_timestep,
                    sd.data,
                    sd.X_buffer,
                    &min_index)
            else:
                min_dist = euclidean_distance(
                    s_offset,
                    s_stride,
                    s_length,
                    s_data,
                    t_offset,
                    sd.timestep_stride,
                    sd.n_timestep,
                    sd.data,
                    &min_index)

            if return_index:
                return min_dist, min_index
            else:
                return min_dist
        else:  # assume an `array_like` object for `samples`
            samples = check_array(sample, ensure_2d=False, dtype=np.int)
            dist = []
            ind = []
            for i in samples:
                check_sample_(i, sd.n_samples)
                t_offset = i * sd.sample_stride + \
                           dim * sd.dim_stride
                if scale:
                    min_dist = scaled_euclidean_distance(
                        s_offset,
                        s_stride,
                        s_length,
                        mean,
                        std,
                        s_data,
                        t_offset,
                        sd.timestep_stride,
                        sd.n_timestep,
                        sd.data,
                        sd.X_buffer,
                        &min_index)
                else:
                    min_dist = euclidean_distance(
                        s_offset,
                        s_stride,
                        s_length,
                        s_data,
                        t_offset,
                        sd.timestep_stride,
                        sd.n_timestep,
                        sd.data,
                        &min_index)
                dist.append(min_dist)
                ind.append(min_index)

            if return_index:
                return np.array(dist), np.array(ind)
            else:
                return np.array(dist)
    finally:
        free_ts_database(sd)


def matches(shapelet, data, threshold, dim=0, sample=None, scale=False):
    """Return the positions in data (one array per `sample`) where
    `shapelet` is closer than `threshold`.

    :param s: the subsequence `array_like`
    :param x: the samples [n_samples, n_timesteps]
    :param threshold: the maximum threshold for match
    :param dim: the time series dimension to search (default: 0)
    :param sample: the samples to compare to `int` or `array_like` or `None`.
                   If `None` compare to all. (default: `None`)
    :param scale: search in scaled space (default: `False`)
    :returns: `[n_matches]`, or `[[n_matches], ... n_samples]`
    """
    cdef np.ndarray s = validate_shapelet_(shapelet)
    cdef np.ndarray x = validate_data_(data)
    check_dim(dim, x.ndim)
    if sample is None:
        if x.shape[0] == 1:
            sample = 0
        else:
            sample = np.arange(x.shape[0])

    cdef TSDatabase sd = new_ts_database(x)

    cdef size_t* matches
    cdef size_t n_matches

    cdef size_t s_offset = 0
    cdef size_t s_stride = <size_t> s.strides[0] // s.itemsize
    cdef size_t s_length = s.shape[0]
    cdef double* s_data = <double*> s.data

    cdef size_t t_offset

    cdef double mean = 0
    cdef double std = 0

    if scale:
        mean = np.mean(s)
        std = np.std(s)

    cdef size_t i
    try:
        if isinstance(sample, int):
            check_sample_(sample, sd.n_samples)
            t_offset = sample * sd.sample_stride + \
                       dim * sd.dim_stride
            if scale:
                scaled_euclidean_distance_matches(
                    s_offset,
                    s_stride,
                    s_length,
                    mean,
                    std,
                    s_data,
                    t_offset,
                    sd.timestep_stride,
                    sd.n_timestep,
                    sd.data,
                    sd.X_buffer,
                    threshold,
                    &matches,
                    &n_matches)
            else:
                euclidean_distance_matches(
                    s_offset,
                    s_stride,
                    s_length,
                    s_data,
                    t_offset,
                    sd.timestep_stride,
                    sd.n_timestep,
                    sd.data,
                    threshold,
                    &matches,
                    &n_matches)
            arr = make_numpy_array_(matches, n_matches)
            free(matches)
            return arr
        else:
            samples = check_array(sample, ensure_2d=False, dtype=np.int)
            indicies = []
            for i in samples:
                check_sample_(i, sd.n_samples)
                t_offset = i * sd.sample_stride + \
                           dim * sd.dim_stride
                if scale:
                    scaled_euclidean_distance_matches(
                        s_offset,
                        s_stride,
                        s_length,
                        mean,
                        std,
                        s_data,
                        t_offset,
                        sd.timestep_stride,
                        sd.n_timestep,
                        sd.data,
                        sd.X_buffer,
                        threshold,
                        &matches,
                        &n_matches)
                else:
                    euclidean_distance_matches(
                        s_offset,
                        s_stride,
                        s_length,
                        s_data,
                        t_offset,
                        sd.timestep_stride,
                        sd.n_timestep,
                        sd.data,
                        threshold,
                        &matches,
                        &n_matches)
                arr = make_numpy_array_(matches, n_matches)
                free(matches)
                indicies.append(arr)
            return indicies

    finally:
        free_ts_database(sd)
