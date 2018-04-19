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
   cdef size_t dim

   cdef double distance(self, const SlidingDistance t, size_t t_index) nogil

   cdef void distances(self,
                       const SlidingDistance t,
                       size_t* samples,
                       size_t n_samples,
                       double* distances) nogil


cdef class ScaledShapelet(Shapelet):
   cdef readonly double mean
   cdef readonly double std


# TODO: consider adding `new_shapelet_info(i, s, l, ts)`
# for computing the mean and std
cdef struct ShapeletInfo:
   size_t index  # the index of the shapelt sample
   size_t start  # the start position
   size_t length # the length of the shapelet
   double mean   # the mean of the shapelet
   double std    # the stanard devision
   size_t dim    # the dimension of the shapelet


cdef struct SlidingDistance:
   size_t n_samples       # the number of samples
   size_t n_timestep      # the number of timesteps
   size_t n_dims

   double* X              # the data
   size_t sample_stride   # the stride for samples
   size_t timestep_stride # the `feature` stride
   size_t dim_stride      # the dimension stride

   double* X_buffer       # buffer for normalization

cdef ShapeletInfo new_shapelet_info(size_t index, size_t start,
                                    size_t length) nogil


cdef int shapelet_info_update_statistics(ShapeletInfo* s,
                                         const SlidingDistance t) nogil


cdef int shapelet_info_scaled_distances(ShapeletInfo s,
                                        const size_t* samples,
                                        size_t n_samples,
                                        const SlidingDistance t,
                                        double* result) nogil


cdef double shapelet_info_scaled_distance(ShapeletInfo s,
                                          const SlidingDistance t,
                                          size_t t_index,
                                          size_t t_dim) nogil


cdef double shapelet_info_distance(ShapeletInfo s,
                                   const SlidingDistance t,
                                   size_t t_index) nogil


cdef int shapelet_info_distances(ShapeletInfo s,
                                 const size_t* samples,
                                 size_t n_samples,
                                 const SlidingDistance t,
                                 double* result) nogil


cdef Shapelet shapelet_info_extract_scaled_shapelet(ShapeletInfo s,
                                                    const SlidingDistance t)


cdef Shapelet shapelet_info_extract_shapelet(ShapeletInfo s,
                                             const SlidingDistance t)

# construct a new sliding distance storage
cdef SlidingDistance new_sliding_distance(np.ndarray X)


cdef int free_sliding_distance(SlidingDistance sd) nogil


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
                                    size_t* index) nogil


cdef double sliding_distance(size_t s_offset,
                             size_t s_stride,
                             size_t s_length,
                             double* S,
                             size_t t_offset,
                             size_t t_stride,
                             size_t t_length,
                             double* T,
                             size_t* index) nogil


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
                                  size_t* n_matches) nogil except -1


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
                                            size_t* n_matches) nogil except -1
