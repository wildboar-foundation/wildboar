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
#
# Authors: Isak Samsten

from libc.stdlib cimport free, malloc

from .._data cimport TSDatabase, ts_database_new
from .._utils cimport RAND_R_MAX, imin, rand_int, shuffle, to_ndarray_double
from ._feature cimport Feature, FeatureEngineer
from .catch22._catch22 cimport _histogram_mode


cdef struct Interval:
    Py_ssize_t start
    Py_ssize_t length
    Py_ssize_t random_output

cdef double _mean(Py_ssize_t stride, double *x, Py_ssize_t length) nogil:
    cdef double v
    cdef Py_ssize_t i
    for i in range(length):
        v += x[i * stride]
    return v / length

cdef double _var(Py_ssize_t stride, double *x, Py_ssize_t length) nogil:
    cdef double mean = _mean(stride, x, length)
    cdef double sum = 0
    cdef double v
    cdef Py_ssize_t i
    for i in range(length):
        v = x[i * stride] - mean
        sum += v * v
    return sum / length

cdef double _slope(Py_ssize_t stride, double *x, Py_ssize_t length) nogil:
    cdef double y_mean = (length + 1) / 2.0
    cdef double x_mean = 0
    cdef double mean_diff = 0
    cdef double mean_y_sqr = 0
    cdef Py_ssize_t i, j

    for i in range(length):
        j = i + 1
        mean_diff += x[stride * i] * j
        x_mean += x[stride * i]
        mean_y_sqr += j * j
    mean_diff /= length
    mean_y_sqr /= length
    x_mean /= length
    return (mean_diff - y_mean * x_mean) / (mean_y_sqr - y_mean ** 2)

cdef class Summarizer:
    cdef void summarize(
            self,
            Py_ssize_t x_stride,
            double *x,
            Py_ssize_t x_length,
            Py_ssize_t out_stride,
            double *out
    ) nogil:
        pass

    cdef Py_ssize_t n_outputs(self) nogil:
        return -1

cdef class MeanSummarizer(Summarizer):
    cdef void summarize(
            self,
            Py_ssize_t x_stride,
            double *x,
            Py_ssize_t length,
            Py_ssize_t out_stride,
            double *out
    ) nogil:
        out[0] = _mean(x_stride, x, length)

    cdef Py_ssize_t n_outputs(self) nogil:
        return 1


cdef class VarianceSummarizer(Summarizer):
    cdef void summarize(
            self,
            Py_ssize_t x_stride,
            double *x,
            Py_ssize_t length,
            Py_ssize_t out_stride,
            double *out
    ) nogil:
        out[0] = _slope(x_stride, x, length)

    cdef Py_ssize_t n_outputs(self) nogil:
        return 1


cdef class SlopeSummarizer(Summarizer):
    cdef void summarize(
            self,
            Py_ssize_t x_stride,
            double *x,
            Py_ssize_t length,
            Py_ssize_t out_stride,
            double *out
    ) nogil:
        out[0] = _slope(x_stride, x, length)

    cdef Py_ssize_t n_outputs(self) nogil:
        return 1


cdef class MultiSummarizer(Summarizer):

    cdef void summarize(
            self,
            Py_ssize_t x_stride,
            double *x,
            Py_ssize_t length,
            Py_ssize_t out_stride,
            double *out
    ) nogil:
        cdef Py_ssize_t i
        for i in range(self.n_outputs()):
            out[i * out_stride] = self._compute(i, x_stride, x, length)

    cdef double _compute(
        self,
        Py_ssize_t measure,
        Py_ssize_t stride,
        double *x,
        Py_ssize_t length
    ) nogil:
        return 0


cdef class MeanVarianceSlopeSummarizer(MultiSummarizer):
    cdef double _compute(
            self,
            Py_ssize_t measure,
            Py_ssize_t stride,
            double *x,
            Py_ssize_t length
    ) nogil:
        cdef Py_ssize_t i
        cdef Py_ssize_t v
        if measure == 0:  # MEAN
            return _mean(stride, x, length)
        elif measure == 1:  # VAR
            return _var(stride, x, length)
        elif measure == 2:  # SLOPE
            return _slope(stride, x, length)

    cdef Py_ssize_t n_outputs(self) nogil:
        return 3


cdef class Catch22Summarizer(MultiSummarizer):
    cdef Py_ssize_t *bin_count
    cdef double *bin_edges

    def __cinit__(self):
        self.bin_count = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * 10)
        self.bin_edges = <double*>malloc(sizeof(double) * 11)

    def __dealloc__(self):
        free(self.bin_count)
        free(self.bin_edges)

    def __reduce__(self):
        return self.__class__, ( )

    cdef double _compute(
            self,
            Py_ssize_t measure,
            Py_ssize_t stride,
            double *x,
            Py_ssize_t length
    ) nogil:
        if measure == 0:
            return _histogram_mode(stride, x, length, self.bin_count, self.bin_edges, 5)
        elif measure == 1:
            return _histogram_mode(stride, x, length, self.bin_count, self.bin_edges, 10)

    cdef Py_ssize_t n_outputs(self) nogil:
        return 2


cdef class IntervalFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t n_intervals
    cdef Summarizer summarizer
    cdef double *out_buffer

    def __cinit__(self, Py_ssize_t n_intervals, Summarizer summarizer, *args, **kwargs):
        self.n_intervals = n_intervals
        self.summarizer = summarizer
        self.out_buffer = <double *> malloc(sizeof(double) * summarizer.n_outputs())

    def __dealloc__(self):
        free(self.out_buffer)

    def __reduce__(self):
        return self.__class__, (self.n_intervals, self.summarizer)

    cdef Py_ssize_t init(self, TSDatabase *td) nogil:
        return 0

    cdef Py_ssize_t get_n_features(self, TSDatabase *td) nogil:
        return td.n_dims * self.n_intervals

    cdef Py_ssize_t get_n_outputs(self, TSDatabase *td) nogil:
        return self.get_n_features(td) * self.summarizer.n_outputs()

    cdef Py_ssize_t next_feature(
            self,
            Py_ssize_t feature_id,
            TSDatabase *td,
            Py_ssize_t *samples,
            Py_ssize_t n_samples,
            Feature *transient,
            size_t *seed,
    ) nogil:
        cdef Interval *interval = <Interval*> malloc(sizeof(Interval))
        interval.length = td.n_timestep // self.n_intervals
        interval.start = (feature_id % self.n_intervals) * interval.length + imin(
            feature_id % self.n_intervals, td.n_timestep % self.n_intervals
        )

        if feature_id % self.n_intervals < td.n_timestep % self.n_intervals:
            interval.length += 1

        interval.random_output = 0
        if self.summarizer.n_outputs() > 1:
            interval.random_output = rand_int(0, self.summarizer.n_outputs(), seed)

        transient.dim = feature_id // self.n_intervals
        transient.feature = interval
        return 0

    cdef Py_ssize_t free_transient_feature(self, Feature *feature) nogil:
        if feature.feature != NULL:
            free(feature.feature)
        return 0

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) nogil:
        if feature.feature != NULL:
            free(feature.feature)
        return 0

    cdef Py_ssize_t init_persistent_feature(
            self,
            TSDatabase *td,
            Feature *transient,
            Feature *persistent
    ) nogil:
        cdef Interval *from_interval = <Interval*> transient.feature
        cdef Interval *to_interval = <Interval*> malloc(sizeof(Interval))
        to_interval.start = from_interval.start
        to_interval.length = from_interval.length
        to_interval.random_output = from_interval.random_output
        persistent.dim = transient.dim
        persistent.feature = to_interval
        return 0

    cdef double transient_feature_value(
            self,
            Feature *feature,
            TSDatabase *td,
            Py_ssize_t sample
    ) nogil:
        cdef Interval *interval = <Interval*> feature.feature
        cdef Py_ssize_t offset = (
                sample * td.sample_stride +
                feature.dim * td.dim_stride +
                interval.start * td.timestep_stride
        )

        self.summarizer.summarize(
            td.timestep_stride,
            td.data + offset,
            interval.length,
            1,
            self.out_buffer,
        )
        return self.out_buffer[interval.random_output]

    cdef double persistent_feature_value(
            self,
            Feature *feature,
            TSDatabase *td,
            Py_ssize_t sample
    ) nogil:
        return self.transient_feature_value(feature, td, sample)

    cdef Py_ssize_t transient_feature_fill(
            self,
            Feature *feature,
            TSDatabase *td,
            Py_ssize_t sample,
            TSDatabase *td_out,
            Py_ssize_t out_sample,
            Py_ssize_t out_feature,
    ) nogil:
        cdef Py_ssize_t n_summarizers = self.summarizer.n_outputs()
        cdef Interval *interval = <Interval*> feature.feature
        cdef Py_ssize_t offset = (
                sample * td.sample_stride +
                feature.dim * td.dim_stride +
                interval.start * td.timestep_stride
        )
        cdef Py_ssize_t out_offset = (
                out_sample * td_out.sample_stride +
                out_feature * n_summarizers * td_out.timestep_stride
        )
        self.summarizer.summarize(
            td.timestep_stride,
            td.data + offset,
            interval.length,
            td_out.timestep_stride,
            td_out.data + out_offset,
        )
        return 0

    cdef Py_ssize_t persistent_feature_fill(
            self,
            Feature *feature,
            TSDatabase *td,
            Py_ssize_t sample,
            TSDatabase *td_out,
            Py_ssize_t out_sample,
            Py_ssize_t out_feature,
    ) nogil:
        return self.transient_feature_fill(
            feature, td, sample, td_out, out_sample, out_feature
        )

    cdef object persistent_feature_to_object(self, Feature *feature):
        cdef Interval *interval = <Interval*> feature.feature
        return feature.dim, (interval.start, interval.length, interval.random_output)

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature):
        dim, (start, length, random_output) = object
        cdef Interval *interval = <Interval*> malloc(sizeof(Interval))
        interval.start = start
        interval.length = length
        interval.random_output = random_output

        feature.dim = dim
        feature.feature = interval
        return 0


cdef class RandomFixedIntervalFeatureEngineer(IntervalFeatureEngineer):

    cdef Py_ssize_t n_random_interval
    cdef Py_ssize_t *random_feature_id

    def __cinit__(
            self,
            Py_ssize_t n_intervals,
            Summarizer summarizer,
            Py_ssize_t n_random_interval
    ):
        self.n_random_interval = n_random_interval
        self.random_feature_id = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * n_intervals)
        cdef Py_ssize_t i
        for i in range(n_intervals):
            self.random_feature_id[i] = i

    cdef Py_ssize_t get_n_features(self, TSDatabase *td) nogil:
        return td.n_dims * self.n_random_interval

    cdef Py_ssize_t next_feature(
            self,
            Py_ssize_t feature_id,
            TSDatabase *td,
            Py_ssize_t *samples,
            Py_ssize_t n_samples,
            Feature *transient,
            size_t *seed,
    ) nogil:
        # reshuffle the feature_ids for each dimension
        if feature_id % self.n_random_interval == 0:
            shuffle(self.random_feature_id, self.n_intervals, seed)

        # we need to rescale the feature_id for the correct dimension
        cdef Py_ssize_t padding = feature_id // self.n_random_interval + 1
        return IntervalFeatureEngineer.next_feature(
            self,
            self.random_feature_id[feature_id] * padding,
            td,
            samples,
            n_samples,
            transient,
            seed
        )

    def __reduce__(self):
        return self.__class__, (self.n_intervals, self.summarizer, self.n_random_interval)


cdef class RandomIntervalFeatureEngineer(IntervalFeatureEngineer):

    cdef Py_ssize_t min_length
    cdef Py_ssize_t max_length

    def __cinit__(
            self,
            Py_ssize_t n_intervals,
            Summarizer summarizer,
            Py_ssize_t min_length,
            Py_ssize_t max_length,
    ):
        self.min_length = min_length
        self.max_length = max_length

    def __reduce__(self):
        return self.__class__, (
            self.n_intervals,
            self.summarizer,
            self.min_length,
            self.max_length,
        )

    cdef Py_ssize_t get_n_features(self, TSDatabase *td) nogil:
        return self.n_intervals

    cdef Py_ssize_t next_feature(
            self,
            Py_ssize_t feature_id,
            TSDatabase *td,
            Py_ssize_t *samples,
            Py_ssize_t n_samples,
            Feature *transient,
            size_t *seed,
    ) nogil:
        cdef Interval *interval = <Interval*> malloc(sizeof(Interval))

        interval.length = rand_int(self.min_length, self.max_length, seed)
        interval.start = rand_int(0, td.n_timestep - interval.length, seed)
        interval.random_output = 0
        if self.summarizer.n_outputs() > 1:
            interval.random_output = rand_int(0, self.summarizer.n_outputs(), seed)

        transient.dim = 0
        if td.n_dims > 1:
            transient.dim = rand_int(1, td.n_dims, seed)

        transient.feature = interval
        return 0