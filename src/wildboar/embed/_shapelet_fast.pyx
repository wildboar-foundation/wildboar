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
#
# Authors: Isak Samsten

from libc.stdlib cimport free, malloc

from .._data cimport TSDatabase
from .._utils cimport RAND_R_MAX, rand_int
from ..distance._distance cimport (
    DistanceMeasure,
    TSCopy,
    TSView,
    get_distance_measure,
    ts_copy_free,
    ts_view_free,
)
from ._feature cimport Feature, FeatureEngineer


cdef class ShapeletFeatureEngineer(FeatureEngineer):

    cdef Py_ssize_t min_shapelet_size
    cdef Py_ssize_t max_shapelet_size

    cdef DistanceMeasure _distance_measure

    def __init__(
        self,
        n_timestep,
        object metric,
        dict metric_params,
        min_shapelet_size,
        max_shapelet_size,
    ):
        self._distance_measure = get_distance_measure(
            n_timestep, metric, metric_params
        )
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size

    @property 
    def distance_measure(self):
        return self._distance_measure
    
    cdef Py_ssize_t init(self, TSDatabase *td) nogil:
        self._distance_measure.init(td)
        return 1

    cdef Py_ssize_t free_transient_feature(self, Feature *feature) nogil:
        if feature.feature != NULL:
            ts_view_free(<TSView*> feature.feature)
            free(feature.feature)

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) nogil:
        cdef TSCopy *ts_copy
        if feature.feature != NULL:
            ts_copy = <TSCopy*> feature.feature
            ts_copy_free(ts_copy)
            free(ts_copy)

    cdef Py_ssize_t init_persistent_feature(
        self, 
        TSDatabase *td,
        Feature *transient, 
        Feature *persistent
    ) nogil:
        cdef TSView *ts_view = <TSView*> transient.feature
        cdef TSCopy *ts_copy = <TSCopy*> malloc(sizeof(TSCopy))
        self._distance_measure.init_ts_copy(ts_copy, ts_view, td)
        persistent.dim = transient.dim
        persistent.feature = ts_copy
        return 1

    cdef double transient_feature_value(
        self,
        Feature *feature,
        TSDatabase *td,
        Py_ssize_t sample
    ) nogil:
        return self._distance_measure.ts_view_sub_distance(
            <TSView*> feature.feature, td, sample
        )

    cdef double persistent_feature_value(
        self,
        Feature *feature,
        TSDatabase *td,
        Py_ssize_t sample
    ) nogil:
        return self._distance_measure.ts_copy_sub_distance(
            <TSCopy*> feature.feature, td, sample
        )

    cdef Py_ssize_t transient_feature_fill(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t sample,
        TSDatabase *td_out,
        Py_ssize_t out_sample,
        Py_ssize_t feature_id,
    ) nogil:
        cdef Py_ssize_t offset = (
            td_out.sample_stride * out_sample + 
            td_out.timestep_stride * feature_id
        )
        td_out.data[offset] = self._distance_measure.ts_view_sub_distance(
            <TSView*> feature.feature, td, sample
        )
        return 0

    cdef Py_ssize_t persistent_feature_fill(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t sample,
        TSDatabase *td_out,
        Py_ssize_t out_sample,
        Py_ssize_t feature_id,
    ) nogil:
        cdef Py_ssize_t offset = (
            td_out.sample_stride * out_sample + 
            td_out.timestep_stride * feature_id
        )
        td_out.data[offset] = self._distance_measure.ts_copy_sub_distance(
            <TSCopy*> feature.feature, td, sample
        )
        return 0

    cdef object persistent_feature_to_object(self, Feature *feature):
        return feature.dim, self._distance_measure.object_from_ts_copy(<TSCopy*>feature.feature)

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature):
        cdef TSCopy *ts_copy = <TSCopy*> malloc(sizeof(TSCopy))
        dim, obj = object
        self._distance_measure.init_ts_copy_from_obj(ts_copy, obj)
        feature.dim = dim
        feature.feature = ts_copy
        return 1

cdef class RandomShapeletFeatureEngineer(ShapeletFeatureEngineer):

    cdef Py_ssize_t n_shapelets

    def __init__(
        self,
        n_timestep,
        object metric,
        dict metric_params,
        min_shapelet_size,
        max_shapelet_size,
        n_shapelets,
    ):
        super().__init__(
            n_timestep,
            metric, 
            metric_params,
            min_shapelet_size,
            max_shapelet_size,
        )
        self.n_shapelets = n_shapelets

    cdef Py_ssize_t get_n_features(self, TSDatabase *td) nogil:
        return self.n_shapelets

    cdef Py_ssize_t next_feature(
        self,
        Py_ssize_t feature_id,
        TSDatabase *td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        Feature *transient,
        size_t *random_seed
    ) nogil:
        if feature_id >= self.n_shapelets:
            return -1
        
        cdef Py_ssize_t shapelet_length
        cdef Py_ssize_t shapelet_start
        cdef Py_ssize_t shapelet_index
        cdef Py_ssize_t shapelet_dim
        cdef TSView *ts_view = <TSView*> malloc(sizeof(TSView))

        shapelet_length = rand_int(
            self.min_shapelet_size, self.max_shapelet_size, random_seed)
        shapelet_start = rand_int(
            0, td.n_timestep - shapelet_length, random_seed)
        shapelet_index = samples[rand_int(0, n_samples, random_seed)]
        if td.n_dims > 1:
            shapelet_dim = rand_int(0, td.n_dims, random_seed)
        else:
            shapelet_dim = 1

        transient.dim = shapelet_dim
        self._distance_measure.init_ts_view(
            td,
            ts_view,
            shapelet_index,
            shapelet_start,
            shapelet_length,
            shapelet_dim,
        )
        transient.feature = ts_view
        return 1


    