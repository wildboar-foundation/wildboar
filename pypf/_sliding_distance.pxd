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

cimport numpy as np

cdef class Shapelet:
   cdef readonly size_t length
   cdef double* data

   cdef double distance(self, const SlidingDistance t, size_t t_index) nogil

   cdef void distances(self,
                       const SlidingDistance t,
                       size_t* samples,
                       size_t n_samples,
                       double* distances) nogil

   cdef size_t closer_than(self,
                           const SlidingDistance t,
                           size_t t_index,
                           double threshold,
                           size_t* matches,
                           double* distances,
                           size_t initial_capacity) nogil

   cdef size_t index_distance(self,
                              const SlidingDistance t,
                              size_t t_index,
                              double* min_dist) nogil

   cdef void index_distances(self,
                             const SlidingDistance t,
                             size_t* samples,
                             size_t n_samples,
                             size_t* min_index,
                             double* min_distance) nogil

# TODO: consider adding `new_shapelet_info(i, s, l, ts)`
cdef struct ShapeletInfo:
   size_t index  # the index of the shapelt sample
   size_t start  # the start position
   size_t length # the length of the shapelet
   double mean   # the mean of the shapelet
   double std    # the stanard devision


cdef struct SlidingDistance:
   size_t n_samples       # the number of samples
   size_t n_timestep      # the number of timesteps

   double* X              # the data
   size_t sample_stride   # the stride for samples
   size_t timestep_stride # the `feature` stride

   double* X_buffer       # buffer for normalization


cdef int shapelet_info_update_statistics(ShapeletInfo* s,
                                         const SlidingDistance t) nogil

cdef int shapelet_info_distances(ShapeletInfo s,
                                 const size_t* samples,
                                 size_t n_samples,
                                 const SlidingDistance t,
                                 double* result) nogil

cdef double shapelet_info_distance(ShapeletInfo s,
                                   const SlidingDistance t,
                                   size_t t_index) nogil

cdef Shapelet shapelet_info_extract_shapelet(ShapeletInfo s,
                                             const SlidingDistance t)


# construct a new sliding distance storage
cdef SlidingDistance new_sliding_distance(
    np.ndarray[np.float64_t, ndim=2, mode="c"] X)

cdef int free_sliding_distance(SlidingDistance sd) nogil
