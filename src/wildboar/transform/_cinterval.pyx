# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cimport numpy as np

import numpy as np

from libc.math cimport NAN
from libc.stdlib cimport free, malloc

from ..utils cimport stats
from ..utils.data cimport Dataset
from ..utils.misc cimport to_ndarray_double
from ..utils.rand cimport RAND_R_MAX, rand_int, shuffle
from ._feature cimport Feature, FeatureEngineer
from .catch22 cimport _catch22


cdef struct Interval:
    Py_ssize_t start
    Py_ssize_t length
    Py_ssize_t random_output


cdef class Summarizer:
    cdef void summarize_all(self, double *x, Py_ssize_t length, double *out) nogil:
        cdef Py_ssize_t i
        for i in range(self.n_outputs()):
            out[i] = self.summarize(i, x, length)

    cdef double summarize(self, Py_ssize_t i, double *x, Py_ssize_t length) nogil:
        pass

    cdef void reset(self, Dataset td) nogil:
        pass

    cdef Py_ssize_t n_outputs(self) nogil:
        return -1


cdef class MeanSummarizer(Summarizer):
    cdef double summarize(self, Py_ssize_t i, double *x, Py_ssize_t length) nogil:
        return stats.mean(x, length)
        
    cdef Py_ssize_t n_outputs(self) nogil:
        return 1


cdef class VarianceSummarizer(Summarizer):
    cdef double summarize(self, Py_ssize_t i, double *x, Py_ssize_t length) nogil:
        return stats.variance(x, length)

    cdef Py_ssize_t n_outputs(self) nogil:
        return 1


cdef class SlopeSummarizer(Summarizer):
    cdef double summarize(self, Py_ssize_t i, double *x, Py_ssize_t length) nogil:
        return stats.slope(x, length)

    cdef Py_ssize_t n_outputs(self) nogil:
        return 1


cdef class MeanVarianceSlopeSummarizer(Summarizer):
    cdef double summarize(self, Py_ssize_t i, double *x, Py_ssize_t length) nogil:
        if i == 0:
            return stats.mean(x, length)
        elif i == 1:
            return stats.variance(x, length)
        elif i == 2:
            return stats.slope(x, length)
        else:
            return NAN

    cdef Py_ssize_t n_outputs(self) nogil:
        return 3


cdef class Catch22Summarizer(Summarizer):
    cdef int *bin_count
    cdef double *bin_edges
    cdef double *ac
    cdef double *welch_f
    cdef double *welch_s
    cdef double *window

    def __cinit__(self):
        self.bin_count = <int*>malloc(sizeof(int) * 10)
        self.bin_edges = <double*>malloc(sizeof(double) * 11)
        self.ac = NULL 
        self.welch_f = NULL 
        self.welch_s = NULL
        self.window = NULL
        if self.bin_count == NULL or self.bin_edges == NULL:
            raise MemoryError()

    cdef void reset(self, Dataset td) nogil:
        if self.ac != NULL:
            free(self.ac)

        if self.welch_s != NULL:
            free(self.welch_s)

        if self.welch_f != NULL:
            free(self.welch_f)

        if self.window != NULL:
            free(self.window)
        
        cdef Py_ssize_t n_timestep = td.n_timestep
        cdef Py_ssize_t welch_length = stats.next_power_of_2(n_timestep)
        self.ac = <double*> malloc(sizeof(double) * n_timestep)
        self.window = <double*> malloc(sizeof(double)* n_timestep);
        self.welch_f = <double*> malloc(sizeof(double) * welch_length )
        self.welch_s = <double*> malloc(sizeof(double) * welch_length )

        if (
            self.ac == NULL or
            self.welch_s == NULL or
            self.welch_f == NULL or
            self.window == NULL
        ):
            raise MemoryError()

        cdef Py_ssize_t i
        for i in range(n_timestep):
            self.window[i] = 1.0

    def __dealloc__(self):
        free(self.bin_count)
        free(self.bin_edges)
        if self.ac != NULL:
            free(self.ac)

        if self.welch_f != NULL:
            free(self.welch_f)

        if self.welch_s != NULL:
            free(self.welch_s)

        if self.window != NULL:
            free(self.window)

    def __reduce__(self):
        return self.__class__, ( )

    # This is ugly
    cdef double summarize(self, Py_ssize_t i, double *x, Py_ssize_t length) nogil:
        cdef Py_ssize_t welch_length = -1
        cdef int n_welch = -1
        
        if i == 15 or i == 20:    
            n_welch = stats.welch(
                x, 
                length, 
                stats.next_power_of_2(length), 
                1.0, 
                self.window, 
                length, 
                self.welch_s, 
                self.welch_f
            )
        if i == 2 or i == 3 or i == 8 or i == 10 or i == 12:
            stats.auto_correlation(x, length, self.ac)

        if i == 0:
            return _catch22.histogram_mode5(x, length, self.bin_count, self.bin_edges)
        elif i == 1:
            return _catch22.histogram_mode10(x, length, self.bin_count, self.bin_edges)
        elif i == 2:
            return _catch22.f1ecac(self.ac, length)
        elif i == 3:
            return _catch22.first_min(self.ac, length)
        elif i == 4:
            return _catch22.histogram_ami_even_2_5(x, length)
        elif i == 5:
            return _catch22.trev_1_num(x, length)
        elif i == 6:
            return _catch22.hrv_classic_pnn(x, length, 40)
        elif i == 7:
            return _catch22.above_mean_stretch(x, length)
        elif i == 8:
            return _catch22.transition_matrix_3ac_sumdiagcov(x, self.ac, length)
        elif i == 9:
            return _catch22.periodicity_wang_th0_01(x, length)
        elif i == 10:
            return _catch22.embed2_dist_tau_d_expfit_meandiff(x, self.ac, length)
        elif i == 11:
            return _catch22.auto_mutual_info_stats_gaussian_fmmi(x, length, 40)
        elif i == 12:
            return _catch22.local_mean_tauresrat(x, self.ac, length, 1)
        elif i == 13:
            return _catch22.outlier_include_np_mdrmd(x, length, 1, 0.01)
        elif i == 14:
            return _catch22.outlier_include_np_mdrmd(x, length, -1, 0.01)
        elif i == 15:
            return _catch22.summaries_welch_rect(x, length, 1, self.welch_s, self.welch_f, n_welch)
        elif i == 16:
            return _catch22.below_diff_stretch(x, length)
        elif i == 17:
            return _catch22.motif_three_quantile_hh(x, length)
        elif i == 18:
            return _catch22.fluct_anal_2_50_1_logi_prop_r1(x, length, 1, 0)
        elif i == 19:
            return _catch22.fluct_anal_2_50_1_logi_prop_r1(x, length, 2, 1)
        elif i == 20:
            return _catch22.summaries_welch_rect(x, length, 0, self.welch_s, self.welch_f, n_welch)
        elif i == 21:
            return _catch22.local_mean_std(x, length, 3)
        else:
            return NAN

    cdef Py_ssize_t n_outputs(self) nogil:
        return 22

cdef class PyFuncSummarizer(Summarizer):
    cdef list func
    cdef np.ndarray x_buffer

    def __cinit__(self, func):
        self.func = func
        self.x_buffer = None

    def __reduce__(self):
        return self.__class__, (self.func, )

    cdef void reset(self, Dataset td) nogil:
        with gil:
            self.x_buffer = np.ndarray(td.n_timestep)

    cdef void summarize_all(
            self,
            double *x,
            Py_ssize_t length,
            double *out
    ) nogil:
        cdef double value
        cdef Py_ssize_t i
        with gil:
            for i in range(length):
                self.x_buffer[i] = x[i]

        for i in range(self.n_outputs()):
            with gil:
                value = self.func[i](self.x_buffer[0:length])
            out[i] = value

    cdef double summarize(self, Py_ssize_t m, double *x, Py_ssize_t length) nogil:
        cdef double value
        cdef Py_ssize_t i
        with gil:
            for i in range(length):
                self.x_buffer[i] = x[i]
            value = self.func[m](self.x_buffer[0:length])
        return value

    cdef Py_ssize_t n_outputs(self) nogil:
        with gil:
            return len(self.func)


cdef inline Py_ssize_t _imin(Py_ssize_t a, Py_ssize_t b) nogil:
    return a if a < b else b


cdef class IntervalFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t n_intervals
    cdef Summarizer summarizer

    def __cinit__(self, Py_ssize_t n_intervals, Summarizer summarizer, *args, **kwargs):
        self.n_intervals = n_intervals
        self.summarizer = summarizer

    def __reduce__(self):
        return self.__class__, (self.n_intervals, self.summarizer)

    cdef Py_ssize_t reset(self, Dataset td) nogil:
        self.summarizer.reset(td)
        return 0

    cdef Py_ssize_t get_n_features(self, Dataset td) nogil:
        return td.n_dims * self.n_intervals

    cdef Py_ssize_t get_n_outputs(self, Dataset td) nogil:
        return self.get_n_features(td) * self.summarizer.n_outputs()

    cdef Py_ssize_t next_feature(
            self,
            Py_ssize_t feature_id,
            Dataset td,
            Py_ssize_t *samples,
            Py_ssize_t n_samples,
            Feature *transient,
            size_t *seed,
    ) nogil:
        cdef Interval *interval = <Interval*> malloc(sizeof(Interval))
        interval.length = td.n_timestep // self.n_intervals
        interval.start = (feature_id % self.n_intervals) * interval.length + _imin(
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
            Dataset td,
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
            Dataset td,
            Py_ssize_t sample
    ) nogil:
        cdef Interval *interval = <Interval*> feature.feature
        return self.summarizer.summarize(
            interval.random_output,
            td.get_sample(sample, feature.dim) + interval.start,
            interval.length,
        )

    cdef double persistent_feature_value(
            self,
            Feature *feature,
            Dataset td,
            Py_ssize_t sample
    ) nogil:
        return self.transient_feature_value(feature, td, sample)

    cdef Py_ssize_t transient_feature_fill(
            self,
            Feature *feature,
            Dataset td,
            Py_ssize_t sample,
            Dataset td_out,
            Py_ssize_t out_sample,
            Py_ssize_t out_feature,
    ) nogil:
        cdef Py_ssize_t n_summarizers = self.summarizer.n_outputs()
        cdef Interval *interval = <Interval*> feature.feature
        self.summarizer.summarize_all(
            td.get_sample(sample, feature.dim) + interval.start,
            interval.length,
            td_out.get_sample(out_sample, 0) + out_feature * n_summarizers,
        )
        return 0

    cdef Py_ssize_t persistent_feature_fill(
            self,
            Feature *feature,
            Dataset td,
            Py_ssize_t sample,
            Dataset td_out,
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

    cdef Py_ssize_t get_n_features(self, Dataset td) nogil:
        return td.n_dims * self.n_random_interval

    cdef Py_ssize_t next_feature(
            self,
            Py_ssize_t feature_id,
            Dataset td,
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

    cdef Py_ssize_t get_n_features(self, Dataset td) nogil:
        return self.n_intervals

    cdef Py_ssize_t next_feature(
            self,
            Py_ssize_t feature_id,
            Dataset td,
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