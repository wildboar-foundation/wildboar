# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from libc.math cimport NAN, pow, ceil, log2, floor
from libc.stdlib cimport free, malloc
from numpy cimport uint32_t

from ..utils cimport TSArray, _stats
from ..utils._misc cimport to_ndarray_double, argsort
from ..utils._rand cimport RAND_R_MAX, rand_int, shuffle
from ._attr_gen cimport Attribute, AttributeGenerator
from .catch22 cimport _catch22


cdef struct Interval:
    Py_ssize_t start
    Py_ssize_t length
    Py_ssize_t random_output


cdef class Summarizer:
    cdef Py_ssize_t n_timestep

    cdef void summarize_all(
        self, const double *x, Py_ssize_t length, double *out, Py_ssize_t n_outputs
    ) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n_outputs):
            out[i] = self.summarize(i, x, length, n_outputs)

    cdef double summarize(
        self, Py_ssize_t i, const double *x, Py_ssize_t length, Py_ssize_t n_outputs
    ) noexcept nogil:
        pass

    cdef void reset(self, TSArray X) noexcept nogil:
        self.n_timestep = X.shape[2]

    cdef Py_ssize_t n_outputs(self, Py_ssize_t length) noexcept nogil:
        return -1

    def value_name(self, Py_ssize_t i):
        return "?"

cdef class QuantSummarizer(Summarizer):
    cdef Py_ssize_t v
    cdef Py_ssize_t *index

    def __cinit__(self, Py_ssize_t v=4):
        self.v = v
        self.index = NULL

    def __dealloc__(self):
        if self.index != NULL:
            free(self.index)
            self.index = NULL

    def __reduce__(self):
        return self.__class__, (self.v, )

    cdef void reset(self, TSArray X) noexcept nogil:
        Summarizer.reset(self, X)
        if self.index != NULL:
            free(self.index)
            self.index = NULL

        self.index = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * X.shape[2])

    cdef Py_ssize_t n_outputs(self, Py_ssize_t length) noexcept nogil:
        return max(1, length // self.v)

    cdef void summarize_all(
        self, const double *x, Py_ssize_t length, double *out, Py_ssize_t n_outputs
    ) noexcept nogil:
        cdef Py_ssize_t i, j
        #cdef double n_outputs = self.n_outputs(length) - 1
        if length > 2:
            for i in range(length):
                self.index[i] = i
            argsort(x, self.index, length)

        for i in range(n_outputs):
            if length == 1:
                out[i] = x[0]
            elif length == 2:
                out[i] = (x[0] + x[1]) / 2
            else:
                j = <Py_ssize_t> floor((i / <float>(n_outputs - 1)) * (length - 1))
                out[i] = x[self.index[j]]

    cdef double summarize(
        self, Py_ssize_t i, const double *x, Py_ssize_t length, Py_ssize_t n_outputs
    ) noexcept nogil:
        cdef Py_ssize_t j

        if length == 1:
            return x[0]
        elif length == 2:
            return (x[0]+x[1]) / 2
        else:
            for j in range(length):
                self.index[j] = j
            argsort(x, self.index, length)

            j = <Py_ssize_t> floor((i / (n_outputs - 1)) * (length - 1))
            return x[self.index[j]]


cdef class MeanSummarizer(Summarizer):
    cdef double summarize(
        self, Py_ssize_t i, const double *x, Py_ssize_t length, Py_ssize_t n_outputs
    ) noexcept nogil:
        return _stats.mean(x, length)

    cdef Py_ssize_t n_outputs(self, Py_ssize_t length) noexcept nogil:
        return 1

    def value_name(self, Py_ssize_t i):
        return "mean"


cdef class VarianceSummarizer(Summarizer):
    cdef double summarize(
        self, Py_ssize_t i, const double *x, Py_ssize_t length, Py_ssize_t n_outputs
    ) noexcept nogil:
        return _stats.variance(x, length)

    cdef Py_ssize_t n_outputs(self, Py_ssize_t length) noexcept nogil:
        return 1

    def value_name(self, Py_ssize_t i):
        return "var"


cdef class SlopeSummarizer(Summarizer):
    cdef double summarize(
        self, Py_ssize_t i, const double *x, Py_ssize_t length, Py_ssize_t n_outputs
    ) noexcept nogil:
        return _stats.slope(x, length)

    cdef Py_ssize_t n_outputs(self, Py_ssize_t length) noexcept nogil:
        return 1

    def value_name(self, Py_ssize_t i):
        return "slope"


cdef class MeanVarianceSlopeSummarizer(Summarizer):
    cdef double summarize(
        self, Py_ssize_t i, const double *x, Py_ssize_t length, Py_ssize_t n_outputs
    ) noexcept nogil:
        if i == 0:
            return _stats.mean(x, length)
        elif i == 1:
            return _stats.variance(x, length)
        elif i == 2:
            return _stats.slope(x, length)
        else:
            return NAN

    cdef Py_ssize_t n_outputs(self, Py_ssize_t length) noexcept nogil:
        return 3

    def value_name(self, Py_ssize_t i):
        if i == 0:
            return "mean"
        elif i == 1:
            return "var"
        elif i == 2:
            return "slope"
        else:
            return "?"


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

    cdef void reset(self, TSArray X) noexcept nogil:
        Summarizer.reset(self, X)
        if self.ac != NULL:
            free(self.ac)

        if self.welch_s != NULL:
            free(self.welch_s)

        if self.welch_f != NULL:
            free(self.welch_f)

        if self.window != NULL:
            free(self.window)

        cdef Py_ssize_t n_timestep = X.shape[2]
        cdef Py_ssize_t welch_length = _stats.next_power_of_2(n_timestep)
        self.ac = <double*> malloc(sizeof(double) * n_timestep)
        self.window = <double*> malloc(sizeof(double)* n_timestep)
        self.welch_f = <double*> malloc(sizeof(double) * welch_length)
        self.welch_s = <double*> malloc(sizeof(double) * welch_length)

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
        return self.__class__, ()

    # This is ugly
    cdef double summarize(
        self, Py_ssize_t i, const double *x, Py_ssize_t length, Py_ssize_t n_outputs
    ) noexcept nogil:
        cdef int n_welch = -1

        if i == 15 or i == 20:
            n_welch = _stats.welch(
                x,
                length,
                _stats.next_power_of_2(length),
                1.0,
                self.window,
                length,
                self.welch_s,
                self.welch_f
            )
        if i == 2 or i == 3 or i == 8 or i == 10 or i == 12:
            _stats.auto_correlation(x, length, self.ac)

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
            return _catch22.summaries_welch_rect(
                x, length, 1, self.welch_s, self.welch_f, n_welch
            )
        elif i == 16:
            return _catch22.below_diff_stretch(x, length)
        elif i == 17:
            return _catch22.motif_three_quantile_hh(x, length)
        elif i == 18:
            return _catch22.fluct_anal_2_50_1_logi_prop_r1(x, length, 1, 0)
        elif i == 19:
            return _catch22.fluct_anal_2_50_1_logi_prop_r1(x, length, 2, 1)
        elif i == 20:
            return _catch22.summaries_welch_rect(
                x, length, 0, self.welch_s, self.welch_f, n_welch
            )
        elif i == 21:
            return _catch22.local_mean_std(x, length, 3)
        else:
            return NAN

    cdef Py_ssize_t n_outputs(self, Py_ssize_t length) noexcept nogil:
        return 22

cdef class PyFuncSummarizer(Summarizer):
    cdef list func
    cdef object x_buffer

    def __cinit__(self, func):
        self.func = func
        self.x_buffer = None

    def __reduce__(self):
        return self.__class__, (self.func, )

    cdef void reset(self, TSArray X) noexcept nogil:
        Summarizer.reset(self, X)
        with gil:
            self.x_buffer = np.empty(X.shape[2], dtype=float)

    cdef void summarize_all(
            self,
            double *x,
            Py_ssize_t length,
            double *out,
            Py_ssize_t n_outputs,
    ) noexcept nogil:
        cdef double value
        cdef Py_ssize_t i
        with gil:
            for i in range(length):
                self.x_buffer[i] = x[i]

        for i in range(n_outputs):
            with gil:
                value = self.func[i](self.x_buffer[0:length])
            out[i] = value

    cdef double summarize(
        self, Py_ssize_t m, const double *x, Py_ssize_t length, Py_ssize_t n_outputs
    ) noexcept nogil:
        cdef double value
        cdef Py_ssize_t i
        with gil:
            for i in range(length):
                self.x_buffer[i] = x[i]
            value = self.func[m](self.x_buffer[0:length])
        return value

    cdef Py_ssize_t n_outputs(self, Py_ssize_t length) noexcept nogil:
        with gil:
            return len(self.func)


cdef inline Py_ssize_t _imin(Py_ssize_t a, Py_ssize_t b) noexcept nogil:
    return a if a < b else b


cdef class IntervalAttributeGenerator(AttributeGenerator):
    cdef readonly Summarizer summarizer

    def __cinit__(self, Summarizer summarizer, *args, **kwargs):
        self.summarizer = summarizer

    cdef int _reset(self, TSArray X) noexcept nogil:
        self.summarizer.reset(X)
        return 0

    cdef Py_ssize_t free_transient(self, Attribute *attribute) noexcept nogil:
        if attribute.attribute != NULL:
            free(attribute.attribute)
        return 0

    cdef Py_ssize_t free_persistent(self, Attribute *attribute) noexcept nogil:
        if attribute.attribute != NULL:
            free(attribute.attribute)
        return 0

    cdef Py_ssize_t init_persistent(
            self,
            TSArray X,
            Attribute *transient,
            Attribute *persistent
    ) noexcept nogil:
        cdef Interval *from_interval = <Interval*> transient.attribute
        cdef Interval *to_interval = <Interval*> malloc(sizeof(Interval))
        to_interval.start = from_interval.start
        to_interval.length = from_interval.length
        to_interval.random_output = from_interval.random_output
        persistent.dim = transient.dim
        persistent.attribute = to_interval
        return 0

    cdef double transient_value(
            self,
            Attribute *attribute,
            TSArray X,
            Py_ssize_t sample
    ) noexcept nogil:
        cdef Interval *interval = <Interval*> attribute.attribute
        return self.summarizer.summarize(
            interval.random_output,
            &X[sample, attribute.dim, interval.start],
            interval.length,
            self.summarizer.n_outputs(interval.length),
        )

    cdef double persistent_value(
            self,
            Attribute *attribute,
            TSArray X,
            Py_ssize_t sample
    ) noexcept nogil:
        return self.transient_value(attribute, X, sample)

    cdef Py_ssize_t transient_fill(
            self,
            Attribute *attribute,
            TSArray X,
            Py_ssize_t sample,
            double[:, :] out,
            Py_ssize_t out_sample,
            Py_ssize_t out_attribute,
    ) noexcept nogil:
        cdef Interval *interval = <Interval*> attribute.attribute
        cdef Py_ssize_t n_outputs = self.summarizer.n_outputs(interval.length)
        self.summarizer.summarize_all(
            &X[sample, attribute.dim, interval.start],
            interval.length,
            &out[out_sample, out_attribute * n_outputs],
            n_outputs,
        )
        return 0

    cdef Py_ssize_t persistent_fill(
            self,
            Attribute *attribute,
            TSArray X,
            Py_ssize_t sample,
            double[:, :] out,
            Py_ssize_t out_sample,
            Py_ssize_t out_attribute,
    ) noexcept nogil:
        return self.transient_fill(
            attribute, X, sample, out, out_sample, out_attribute
        )

    cdef object persistent_to_object(self, Attribute *attribute):
        cdef Interval *interval = <Interval*> attribute.attribute
        return attribute.dim, (interval.start, interval.length, interval.random_output)

    cdef Py_ssize_t persistent_from_object(self, object object, Attribute *attribute):
        dim, (start, length, random_output) = object
        cdef Interval *interval = <Interval*> malloc(sizeof(Interval))
        interval.start = start
        interval.length = length
        interval.random_output = random_output

        attribute.dim = dim
        attribute.attribute = interval
        return 0


cdef class FixedIntervalAttributeGenerator(IntervalAttributeGenerator):

    cdef Py_ssize_t n_intervals

    def __cinit__(self, Summarizer summarizer, Py_ssize_t n_intervals, *args, **kwargs):
        self.n_intervals = n_intervals

    def __reduce__(self):
        return self.__class__, (self.summarizer, self.n_intervals)

    cdef Py_ssize_t get_n_attributes(
        self, Py_ssize_t* samples, Py_ssize_t n_samples
    ) noexcept nogil:
        return self.n_dims * self.n_intervals

    cdef Py_ssize_t get_n_outputs(
        self, Py_ssize_t *samples, Py_ssize_t n_samples
    ) noexcept nogil:
        return (
            self.get_n_attributes(samples, n_samples) *
            self.summarizer.n_outputs(self.n_timestep // self.n_intervals)
        )

    cdef Py_ssize_t next_attribute(
            self,
            Py_ssize_t attribute_id,
            TSArray X,
            Py_ssize_t *samples,
            Py_ssize_t n_samples,
            Attribute *transient,
            uint32_t *seed,
    ) noexcept nogil:
        cdef Interval *interval = <Interval*> malloc(sizeof(Interval))
        interval.length = X.shape[2] // self.n_intervals
        interval.start = (attribute_id % self.n_intervals) * interval.length + _imin(
            attribute_id % self.n_intervals, X.shape[2] % self.n_intervals
        )

        if attribute_id % self.n_intervals < X.shape[2] % self.n_intervals:
            interval.length += 1

        interval.random_output = 0
        if self.summarizer.n_outputs(interval.length) > 1:
            interval.random_output = rand_int(0, self.summarizer.n_outputs(interval.length), seed)

        transient.dim = attribute_id // self.n_intervals
        transient.attribute = interval
        return 0


cdef class RandomFixedIntervalAttributeGenerator(FixedIntervalAttributeGenerator):

    cdef Py_ssize_t n_random_interval
    cdef Py_ssize_t *random_attribute_id

    def __cinit__(
            self,
            Summarizer summarizer,
            Py_ssize_t n_intervals,
            Py_ssize_t n_random_interval
    ):
        self.n_random_interval = n_random_interval
        self.random_attribute_id = <Py_ssize_t*> malloc(
            sizeof(Py_ssize_t) * n_intervals
        )
        cdef Py_ssize_t i
        for i in range(n_intervals):
            self.random_attribute_id[i] = i

    def __dealloc__(self):
        if self.random_attribute_id != NULL:
            free(self.random_attribute_id)
            self.random_attribute_id = NULL

    cdef Py_ssize_t get_n_attributes(
        self, Py_ssize_t* samples, Py_ssize_t n_samples
    ) noexcept nogil:
        return self.n_dims * self.n_random_interval

    cdef Py_ssize_t next_attribute(
            self,
            Py_ssize_t attribute_id,
            TSArray X,
            Py_ssize_t *samples,
            Py_ssize_t n_samples,
            Attribute *transient,
            uint32_t *seed,
    ) noexcept nogil:
        # reshuffle the attribute_ids for each dimension
        if attribute_id % self.n_random_interval == 0:
            shuffle(self.random_attribute_id, self.n_intervals, seed)

        # we need to rescale the attribute_id for the correct dimension
        cdef Py_ssize_t padding = attribute_id // self.n_random_interval + 1
        return FixedIntervalAttributeGenerator.next_attribute(
            self,
            self.random_attribute_id[attribute_id] * padding,
            X,
            samples,
            n_samples,
            transient,
            seed
        )

    def __reduce__(self):
        return self.__class__, (
            self.n_intervals,
            self.summarizer,
            self.n_random_interval
        )


cdef class RandomIntervalAttributeGenerator(FixedIntervalAttributeGenerator):

    cdef Py_ssize_t min_length
    cdef Py_ssize_t max_length

    def __cinit__(
            self,
            Summarizer summarizer,
            Py_ssize_t n_intervals,
            Py_ssize_t min_length,
            Py_ssize_t max_length,
    ):
        self.min_length = min_length
        self.max_length = max_length

    def __reduce__(self):
        return self.__class__, (
            self.summarizer,
            self.n_intervals,
            self.min_length,
            self.max_length,
        )

    cdef Py_ssize_t get_n_attributes(
        self, Py_ssize_t* samples, Py_ssize_t n_samples
    ) noexcept nogil:
        return self.n_intervals

    cdef Py_ssize_t get_n_outputs(
        self, Py_ssize_t *samples, Py_ssize_t n_samples
    ) noexcept nogil:
        return (
            self.get_n_attributes(samples, n_samples) *
            self.summarizer.n_outputs(self.min_length)
        )

    cdef Py_ssize_t transient_fill(
            self,
            Attribute *attribute,
            TSArray X,
            Py_ssize_t sample,
            double[:, :] out,
            Py_ssize_t out_sample,
            Py_ssize_t out_attribute,
    ) noexcept nogil:
        cdef Interval *interval = <Interval*> attribute.attribute
        cdef Py_ssize_t n_outputs = self.summarizer.n_outputs(self.min_length)
        self.summarizer.summarize_all(
            &X[sample, attribute.dim, interval.start],
            interval.length,
            &out[out_sample, out_attribute * n_outputs],
            n_outputs,
        )
        return 0

    cdef double transient_value(
            self,
            Attribute *attribute,
            TSArray X,
            Py_ssize_t sample
    ) noexcept nogil:
        cdef Interval *interval = <Interval*> attribute.attribute
        return self.summarizer.summarize(
            interval.random_output,
            &X[sample, attribute.dim, interval.start],
            interval.length,
            self.summarizer.n_outputs(self.min_length),
        )

    cdef Py_ssize_t next_attribute(
            self,
            Py_ssize_t attribute_id,
            TSArray X,
            Py_ssize_t *samples,
            Py_ssize_t n_samples,
            Attribute *transient,
            uint32_t *seed,
    ) noexcept nogil:
        cdef Interval *interval = <Interval*> malloc(sizeof(Interval))

        interval.length = rand_int(self.min_length, self.max_length, seed)
        interval.start = rand_int(0, X.shape[2] - interval.length, seed)
        interval.random_output = 0
        if self.summarizer.n_outputs(interval.length) > 1:
            interval.random_output = rand_int(0, self.summarizer.n_outputs(interval.length), seed)

        transient.dim = 0
        if X.shape[1] > 1:
            transient.dim = rand_int(1, X.shape[1], seed)

        transient.attribute = interval
        return 0


cdef Py_ssize_t binsearch_depth(Py_ssize_t i) noexcept nogil:
    cdef Py_ssize_t low = 0
    cdef Py_ssize_t high = <Py_ssize_t> ceil(log2(i + 2))  # upper bound estimate
    cdef Py_ssize_t mid

    while low < high:
        mid = (low + high) // 2
        if <Py_ssize_t> pow(2, mid + 1) - 2 - mid > i:
            high = mid
        else:
            low = mid + 1
    return low


# Here n_intervals represents the depth
cdef class DyadicIntervalAttributeGenerator(IntervalAttributeGenerator):

    cdef Py_ssize_t depth
    cdef Py_ssize_t n_first_outputs
    cdef Py_ssize_t n_second_outputs

    def __cinit__(self, Summarizer summarizer, Py_ssize_t depth):
        self.depth = depth

    def __reduce__(self):
        return self.__class__, (self.summarizer, self.depth)

    cdef Py_ssize_t get_n_attributes(
        self, Py_ssize_t *samples, Py_ssize_t n_samples
    ) noexcept nogil:
        cdef Py_ssize_t n_intervals = <Py_ssize_t> pow(2, self.depth) - 1
        return n_intervals + n_intervals - self.depth

    cdef Py_ssize_t get_n_outputs(
        self, Py_ssize_t *samples, Py_ssize_t n_samples
    ) noexcept nogil:
        return self.n_first_outputs + self.n_second_outputs

    cdef int _reset(self, TSArray X) noexcept nogil:
        IntervalAttributeGenerator._reset(self, X)
        self.n_first_outputs = 0
        self.n_second_outputs = 0

        cdef Py_ssize_t i
        cdef Py_ssize_t n_intervals
        for i in range(self.depth):
            n_intervals = <Py_ssize_t>pow(2, i)
            self.n_first_outputs += self.summarizer.n_outputs(self.n_timestep // n_intervals) * n_intervals
            if i > 0:
                self.n_second_outputs += self.summarizer.n_outputs(self.n_timestep // n_intervals) * (n_intervals - 1)

        return 0

    cdef Py_ssize_t transient_fill(
            self,
            Attribute *attribute,
            TSArray X,
            Py_ssize_t sample,
            double[:, :] out,
            Py_ssize_t out_sample,
            Py_ssize_t out_attribute,
    ) noexcept nogil:
        cdef Interval *interval = <Interval*> attribute.attribute
        cdef Py_ssize_t n_first = <Py_ssize_t> pow(2, self.depth) - 1
        cdef bint is_shifted = out_attribute >= n_first

        cdef Py_ssize_t current_depth
        if is_shifted:
            # If the interval is shifted, we need to search for the
            # current depth. Subtracting the n_first intervals
            # from the index, out_attribute is between 0 and n_second
            current_depth = binsearch_depth(out_attribute - n_first)
        else:
            current_depth = <Py_ssize_t> floor(log2(out_attribute + 1))

        cdef Py_ssize_t n_intervals = <Py_ssize_t> pow(2, current_depth)
        cdef Py_ssize_t length = X.shape[2] // n_intervals

        # To ensure that the intervals are in order, we shift the output
        # index so that it is between 0 and n_intervals
        cdef Py_ssize_t shifted_out_attribute

        if is_shifted:
            # If the interval is shifted, we have one fewer intervals
            # and the current depth is, for the position in the output
            # array, shifted one down (depth=0 has no shifted intervals).
            n_intervals -= 1
            current_depth -= 1
            shifted_out_attribute = out_attribute - n_first - (n_intervals - current_depth - 1)
        else:
            shifted_out_attribute = out_attribute - (n_intervals - 1)

        cdef Py_ssize_t n_outputs = self.summarizer.n_outputs(length)
        cdef Py_ssize_t offset = shifted_out_attribute * n_outputs

        cdef Py_ssize_t min_depth = 0
        if is_shifted:
            # If the interval is shifted, we need to move the output index
            # to start after the non-shifted intervals. We also ensure that
            # the minimum depth start at 1 instead of 0.
            offset += self.n_first_outputs
            min_depth = 1

        cdef Py_ssize_t prev_depth
        cdef Py_ssize_t prev_n_intervals

        # Move the offset to the correct depth.
        for prev_depth in range(min_depth, current_depth + min_depth):
            prev_n_intervals = <Py_ssize_t> pow(2, prev_depth)
            offset += (
                self.summarizer.n_outputs(self.n_timestep // prev_n_intervals) *
                (prev_n_intervals - min_depth)
            )

        self.summarizer.summarize_all(
            &X[sample, attribute.dim, interval.start],
            interval.length,
            &out[out_sample, offset],
            self.summarizer.n_outputs(length),
        )
        return 0

    cdef Py_ssize_t next_attribute(
        self,
        Py_ssize_t attribute_id,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Attribute *transient,
        uint32_t *seed
    ) noexcept nogil:
        cdef Py_ssize_t n_first = <Py_ssize_t> pow(2, self.depth) - 1
        cdef bint is_shifted = attribute_id >= n_first

        cdef Py_ssize_t d
        if is_shifted:
            d = binsearch_depth(attribute_id - n_first)
        else:
            d = <Py_ssize_t> floor(log2(attribute_id + 1))

        cdef Py_ssize_t n_intervals = <Py_ssize_t> pow(2, d)
        cdef Py_ssize_t length = X.shape[2] // n_intervals
        cdef Py_ssize_t pad = 0
        cdef Py_ssize_t shifted_attribute_id

        if is_shifted:
            shifted_attribute_id = attribute_id - n_first - (n_intervals - 1 - d)
            n_intervals -= 1
            pad = length // 2
        else:
            shifted_attribute_id = attribute_id - (n_intervals - 1)

        cdef Interval *interval = <Interval*> malloc(sizeof(Interval))
        interval.start = (
            pad +
            (shifted_attribute_id % n_intervals) *
            length +
            _imin(
                shifted_attribute_id % n_intervals,
                (X.shape[2] - pad) % n_intervals
            )
        )

        if shifted_attribute_id % n_intervals < (X.shape[2] - pad) % n_intervals:
            length += 1

        interval.length = length

        interval.random_output = 0
        if self.summarizer.n_outputs(length) > 1:
            interval.random_output = rand_int(0, self.summarizer.n_outputs(length), seed)

        # TODO: this is incorrect since we have shifted the attribute_id
        #       we should introduce a new variable shifted_attribute_id to make
        #       this work. It also requires that we take it into account when
        #       computing the attributes so we know the depth etc.
        transient.dim = 0  # attribute_id // (n_first + n_last)
        transient.attribute = interval
        return 1
