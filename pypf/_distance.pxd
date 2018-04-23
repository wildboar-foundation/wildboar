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

# for computing the mean and std
cdef struct ShapeletInfo:
    size_t index  # the index of the shapelet sample
    size_t start  # the start position
    size_t length # the length of the shapelet
    double mean   # the mean of the shapelet
    double std    # the stanard devision
    size_t dim    # the dimension of the shapelet


cdef struct TSDatabase:
    size_t n_samples       # the number of samples
    size_t n_timestep      # the number of timesteps
    size_t n_dims

    double* data         # the data
    size_t sample_stride   # the stride for samples
    size_t timestep_stride # the `feature` stride
    size_t dim_stride      # the dimension stride

    double* X_buffer       # buffer for normalization


cdef class Shapelet:
    cdef readonly size_t length
    cdef double* data
    cdef size_t dim

    cdef double distance(self, const TSDatabase t, size_t t_index) nogil

    cdef void distances(self,
                        const TSDatabase t,
                        size_t* samples,
                        size_t n_samples,
                        double* distances) nogil


cdef class ScaledShapelet(Shapelet):
    cdef readonly double mean
    cdef readonly double std


cdef class DistanceMeasure:
    cdef TSDatabase td

    cdef void init(self, TSDatabase td) nogil

    cdef ShapeletInfo new_shapelet_info(self,
                                        size_t index,
                                        size_t dim,
                                        size_t start,
                                        size_t length) nogil

    cdef Shapelet new_shapelet(self, ShapeletInfo s)
        
    cdef double distance(self, ShapeletInfo s, size_t t_index) nogil

    cdef void distances(self,
                        ShapeletInfo s,
                        size_t* samples,
                        double* distances,
                        size_t n_samples) nogil

cdef class ScaledDistanceMeasure(DistanceMeasure):
    pass


cdef int shapelet_info_update_statistics(ShapeletInfo* s,
                                         const TSDatabase t) nogil


cdef TSDatabase new_ts_database(np.ndarray X)


cdef int free_ts_database(TSDatabase sd) nogil
