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

from libc.stdlib cimport malloc
from libc.stdlib cimport free

from pypf._sliding_distance cimport SlidingDistance
from pypf._sliding_distance cimport Shapelet
from pypf._sliding_distance cimport ScaledShapelet
from pypf._sliding_distance cimport new_sliding_distance
from pypf._sliding_distance cimport free_sliding_distance

from sklearn.utils import check_array

cdef _make_shapelet(s, normalize):
    cdef size_t i
    cdef Shapelet shapelet
    if isinstance(s, Shapelet):
        shapelet = <Shapelet> s
    else:
        s = check_array(s, ensure_2d=False, dtype=np.float64)
        if normalize:
            std = np.std(s)
            mean = np.mean(s) # refactor
            if std > 0:
                s = (s - mean) / std
            else:
                s = np.zeros(s.shape)
            shapelet = ScaledShapelet(s.shape[0], mean, std)
        else:
            shapelet = Shapelet(s.shape[0])
        
        for i in range(<size_t> s.shape[0]):
            shapelet.data[i] = s[i]
    return shapelet

def min_distance(s, x, sample=None, scale=True, return_index=False):
    """Computes the minimum distance between `s` and the samples in `x`

    :param s: the subsequence `array_like` or `Shapelet`
    :param x: the samples [n_samples, n_timesteps]
    :param sample: the samples to compare to `int` or `array_like` or `None`.
                   If `None` compare to all. (default: None)
    :param scale: scale the shapelet
    :param return_index: if `true` return the first index of the best match
    :returns: `float`,
              `(float, int)`,
              `float [n_samples]` or
              `(float [n_samples], int [n_samples]` depending on input
    """
    x = check_array(x, ensure_2d=False, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, x.shape[0])
    x = np.ascontiguousarray(x)

    if sample == None:
        if x.shape[0] == 1:
            sample = 0
        else:
            sample = np.arange(x.shape[0])

    cdef Shapelet shapelet = _make_shapelet(s, scale)
    cdef SlidingDistance sd = new_sliding_distance(x)
    cdef double min_dist
    cdef size_t min_index

    cdef np.ndarray[np.intp_t] samples
    cdef np.ndarray[np.float64_t] min_distances
    cdef np.ndarray[np.intp_t] min_indicies

    try:
        if isinstance(sample, int):
            if sample > x.shape[0] or sample < 0:
                raise ValueError("illegal sample {}".format(sample))

            if return_index:
                min_index = shapelet.index_distance(sd, sample, &min_dist)
                return min_dist, min_index
            else:
                return shapelet.distance(sd, sample)
        else:  # assume an `array_like` object
            samples = np.asarray(sample)
            check_array(samples, ensure_2d=False, dtype=np.int)
            if samples.ndim != 1 or samples.strides[0] != samples.itemsize:
                raise ValueError("1d-array array expected with stride 1")

            # TODO: consider cython.parallel.prange for speed
            min_distances = np.empty(samples.shape[0], dtype=np.float64)
            if return_index:
                min_indicies= np.empty(samples.shape[0], dtype=np.intp)
                shapelet.index_distances(
                    sd, <size_t*> samples.data, <size_t> samples.shape[0],
                    <size_t*> min_indicies.data, <double*> min_distances.data)
                return min_distances, min_indicies
            else:
                shapelet.distances(
                    sd, <size_t*> samples.data, <size_t> samples.shape[0],
                    <double*> min_distances.data)
                return min_distances

    finally:
        free_sliding_distance(sd)


cdef object _make_numpy_arrays(size_t* matches,
                               double* distances,
                               size_t n_matches):
    if n_matches > 0:
        match_array = np.empty(n_matches, dtype=np.intp)
        distance_array = np.empty(n_matches)
        for i in range(n_matches):
            match_array[i] = matches[i]
            distance_array[i] = distances[i]
        return distance_array, match_array
    else:
        return None, None


def matches(s, x, threshold, sample=None, scale=True,
            initial_capacity=5, return_distances=True):
    if initial_capacity < 1:
        raise ValueError("initial capacity {} < 1".format(initial_capacity))

    x = check_array(x, ensure_2d=False, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, x.shape[0])
    x = np.ascontiguousarray(x)

    if sample == None:
        if x.shape[0] == 1:
            sample = 0
        else:
            sample = np.arange(x.shape[0])

    cdef Shapelet shapelet = _make_shapelet(s, scale)
    cdef SlidingDistance sd = new_sliding_distance(x)

    cdef size_t* matches = <size_t*> malloc(
        sizeof(size_t) * initial_capacity)
    cdef double* distances = <double*> malloc(
        sizeof(double) * initial_capacity)

    cdef size_t n_matches
    cdef size_t i
    try:
        if isinstance(sample, int):
            n_matches = shapelet.closer_than(
                sd, sample, threshold, matches, distances, initial_capacity)
            distance_array, match_array = _make_numpy_arrays(
                matches, distances, n_matches)
            if return_distances:
                return distance_array, match_array
            else:
                return match_array
        else:
            samples = check_array(
                np.asarray(sample), ensure_2d=False, dtype=np.int)
            index_matches = []
            distance_matches = []
            for i in range(<size_t>samples.shape[0]):
                n_matches = shapelet.closer_than(
                    sd, samples[i], threshold, matches,
                    distances, initial_capacity)
                distance_array, match_array = _make_numpy_arrays(
                    matches, distances, n_matches)
                index_matches.append(match_array)
                distance_matches.append(distance_array)

            if return_distances:
                return distance_matches, index_matches
            else:
                mask = np.zeros(
                    [samples.shape[0], sd.n_timestep], dtype=np.bool)
                for i in range(<size_t>samples.shape[0]):
                    j = samples[i]
                    if index_matches[i] is not None:
                        for index in index_matches[i]:
                            mask[j, index:(index + shapelet.length)] = True
                return mask
    finally:
        free(matches)
        free(distances)
        free_sliding_distance(sd)

from pypf._sliding_distance cimport shapelet_info_unscaled_distance
from pypf._sliding_distance cimport shapelet_info_unscaled_distances
from pypf._sliding_distance cimport ShapeletInfo, SlidingDistance

def test(x):
    print(x)
    cdef SlidingDistance sd = new_sliding_distance(x)
    cdef ShapeletInfo s
    s.index = 0
    s.start = 0
    s.length = 3
    print(x[0, 0:3])
    cdef np.ndarray[np.intp_t] i = np.arange(10)
    cdef np.ndarray[np.float64_t] d = np.zeros(10, dtype=np.float64)
    shapelet_info_unscaled_distances(
        s, <size_t*> i.data, i.shape[0], sd, <double*> d.data)
    print(d)
