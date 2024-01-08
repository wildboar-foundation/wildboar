# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

cimport numpy as np
import numpy as np

from libc.math cimport NAN, sqrt, floor, INFINITY
from libc.stdlib cimport free, malloc, labs
from numpy cimport float64_t, intp_t, ndarray, double_t

from ..utils cimport _stats
from ..utils._misc cimport Heap, HeapElement

from copy import deepcopy

from ..utils cimport TSArray
from ..utils._misc cimport List
from ..utils._stats cimport (
    fast_mean_std,
    IncStats,
    inc_stats_init,
    inc_stats_add,
    inc_stats_remove,
    inc_stats_mean,
    inc_stats_variance,
)

from ..utils._parallel import run_in_parallel
from ..utils.validation import check_array


cdef class MetricList(List):

    cdef int reset(self, Py_ssize_t metric, TSArray X, TSArray Y) noexcept nogil:
        return (<Metric> self.get(metric)).reset(X, Y)

    cdef double distance(
        self,
        Py_ssize_t metric,
        TSArray X,
        Py_ssize_t x_index,
        TSArray Y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) noexcept nogil:
        return (<Metric> self.get(metric)).distance(X, x_index, Y, y_index, dim)

    cdef double _distance(
        self,
        Py_ssize_t metric,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return (<Metric> self.get(metric))._distance(x, x_len, y, y_len)


cdef class SubsequenceMetricList(List):

    cdef int reset(self, Py_ssize_t metric, TSArray X) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).reset(X)

    cdef int init_transient(
        self,
        Py_ssize_t metric,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).init_transient(
            X, v, index, start, length, dim
        )

    cdef int init_persistent(
        self,
        Py_ssize_t metric,
        TSArray X,
        SubsequenceView* v,
        Subsequence* s,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).init_persistent(X, v, s)

    cdef int free_transient(self, Py_ssize_t metric, SubsequenceView *t) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).free_transient(t)

    cdef int free_persistent(self, Py_ssize_t metric, Subsequence *t) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).free_persistent(t)

    cdef int from_array(
        self,
        Py_ssize_t metric,
        Subsequence *s,
        object obj,
    ):
        return (<SubsequenceMetric> self.get(metric)).from_array(s, obj)

    cdef object to_array(
        self,
        Py_ssize_t metric,
        Subsequence *s
    ):
        return (<SubsequenceMetric> self.get(metric)).to_array(s)

    cdef double transient_distance(
        self,
        Py_ssize_t metric,
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).transient_distance(
            v, X, index, return_index
        )

    cdef double persistent_distance(
        self,
        Py_ssize_t metric,
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).persistent_distance(
            s, X, index, return_index
        )

    cdef Py_ssize_t transient_matches(
        self,
        Py_ssize_t metric,
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double *distances,
        Py_ssize_t *indices,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).transient_matches(
            v, X, index, threshold, distances, indices
        )

    cdef Py_ssize_t persistent_matches(
        self,
        Py_ssize_t metric,
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double *distances,
        Py_ssize_t *indices,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).persistent_matches(
            s, X, index, threshold, distances, indices
        )


cdef int _ts_view_update_statistics(
    SubsequenceView *v, const double* sample
) noexcept nogil:
    """Update the mean and standard deviation of a shapelet info struct """
    cdef double ex = 0
    cdef double ex2 = 0
    cdef Py_ssize_t i
    for i in range(v.length):
        current_value = sample[i]
        ex += current_value
        ex2 += current_value ** 2

    v.mean = ex / v.length
    ex2 = ex2 / v.length - v.mean * v.mean
    if ex2 > EPSILON:
        v.std = sqrt(ex2)
    else:
        v.std = 0.0
    return 0


cdef class SubsequenceMetric:

    cdef int reset(self, TSArray X) noexcept nogil:
        pass

    cdef int init_transient(
        self,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) noexcept nogil:
        v.index = index
        v.dim = dim
        v.start = start
        v.length = length
        v.mean = NAN
        v.std = NAN
        v.extra = NULL
        return 0

    cdef int init_persistent(
        self, TSArray X, SubsequenceView* v, Subsequence* s
    ) noexcept nogil:
        s.dim = v.dim
        s.length = v.length
        s.mean = v.mean
        s.std = v.std
        s.ts_start = v.start
        s.ts_index = v.index
        s.extra = NULL
        s.data = <double*> malloc(sizeof(double) * v.length)
        if s.data == NULL:
            return -1

        cdef const double *sample = &X[v.index, v.dim, v.start]
        cdef Py_ssize_t i
        for i in range(v.length):
            s.data[i] = sample[i]
        return 0

    cdef int free_transient(self, SubsequenceView *v) noexcept nogil:
        return 0

    cdef int free_persistent(self, Subsequence *v) noexcept nogil:
        if v.data != NULL:
            free(v.data)
            v.data = NULL
        return 0

    cdef int from_array(self, Subsequence *v, object obj):
        dim, arr = obj
        v.dim = dim
        v.length = arr.shape[0]
        v.mean = NAN
        v.std = NAN
        v.data = <double*> malloc(v.length * sizeof(double))
        v.extra = NULL
        if v.data == NULL:
            return -1

        cdef Py_ssize_t i
        for i in range(v.length):
            v.data[i] = arr[i]
        return 0

    cdef object to_array(self, Subsequence *s):
        cdef Py_ssize_t j
        cdef ndarray[float64_t] arr = np.empty(s.length, dtype=float)
        with nogil:
            for j in range(s.length):
                arr[j] = s.data[j]

        return (s.dim, arr)

    cdef double transient_distance(
        self,
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return self._distance(
            &X[v.index, v.dim, v.start],
            v.length,
            v.mean,
            v.std if v.std != 0 else 1.0,
            v.extra,
            &X[index, v.dim, 0],
            X.shape[2],
            return_index,
        )

    cdef double persistent_distance(
        self,
        Subsequence *v,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return self._distance(
            v.data,
            v.length,
            v.mean,
            v.std if v.std != 0 else 1.0,
            v.extra,
            &X[index, v.dim, 0],
            X.shape[2],
            return_index,
        )

    cdef Py_ssize_t transient_matches(
        self,
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return self._matches(
            &X[v.index, v.dim, v.start],
            v.length,
            v.mean,
            v.std if v.std != 0 else 1.0,
            v.extra,
            &X[index, v.dim, 0],
            X.shape[2],
            threshold,
            distances,
            indicies,
        )

    cdef Py_ssize_t persistent_matches(
        self,
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return self._matches(
            s.data,
            s.length,
            s.mean,
            s.std if s.std != 0 else 1.0,
            s.extra,
            &X[index, s.dim, 0],
            X.shape[2],
            threshold,
            distances,
            indicies,
        )

    cdef void transient_profile(
        self,
        SubsequenceView *s,
        TSArray x,
        Py_ssize_t i,
        double *dp,
    ) noexcept nogil:
        self._distance_profile(
            &x[s.index, s.dim, s.start],
            s.length,
            s.mean,
            s.std if s.std != 0 else 1.0,
            s.extra,
            &x[i, s.dim, 0],
            x.shape[2],
            dp,
        )

    cdef void persistent_profile(
        self,
        Subsequence *s,
        TSArray x,
        Py_ssize_t i,
        double *dp,
    ) noexcept nogil:
        self._distance_profile(
            s.data,
            s.length,
            s.mean,
            s.std if s.std != 0 else 1.0,
            s.extra,
            &x[i, s.dim, 0],
            x.shape[2],
            dp,
        )

    cdef void _distance_profile(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        double *dp,
    ) noexcept nogil:
        self._matches(
                s,
                s_len,
                s_mean,
                s_std,
                s_extra,
                x,
                x_len,
                INFINITY,
                dp,
                NULL,
            )

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return NAN

    cdef Py_ssize_t _matches(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        return 0


cdef class ScaledSubsequenceMetric(SubsequenceMetric):

    cdef int init_transient(
        self,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) noexcept nogil:
        cdef int err = SubsequenceMetric.init_transient(
            self, X, v, index, start, length, dim
        )
        if err < 0:
            return err
        _ts_view_update_statistics(v, &X[v.index, v.dim, v.start])
        return 0

    cdef int from_array(
        self,
        Subsequence *v,
        object obj,
    ):
        cdef int err = SubsequenceMetric.from_array(self, v, obj)
        if err == -1:
            return -1

        dim, arr = obj
        v.mean = np.mean(arr)
        v.std = np.std(arr)
        if v.std <= EPSILON:
            v.std = 0.0
        return 0


cdef class ScaledSubsequenceMetricWrap(ScaledSubsequenceMetric):
    cdef Metric wrap
    cdef double *x_buffer
    cdef double *s_buffer

    def __cinit__(self, Metric wrap):
        self.wrap = wrap
        self.x_buffer = NULL
        self.s_buffer = NULL

    def __reduce__(self):
        return self.__class__, (self.wrap, )

    def __dealloc__(self):
        free(self.x_buffer)
        free(self.s_buffer)

    cdef int reset(self, TSArray X) noexcept nogil:
        if self.x_buffer != NULL:
            free(self.x_buffer)
        if self.s_buffer != NULL:
            free(self.s_buffer)

        self.x_buffer = <double*> malloc(sizeof(double) * X.shape[2])
        self.s_buffer = <double*> malloc(sizeof(double) * X.shape[2])
        return self.wrap.reset(X, X)

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        cdef double mean, std
        cdef double min_dist = INFINITY
        cdef Py_ssize_t i, j

        cdef IncStats stats
        inc_stats_init(&stats)

        for i in range(s_len - 1):
            self.s_buffer[i] = (s[i] - s_mean) / s_std
            inc_stats_add(&stats, 1.0, x[i])

        self.s_buffer[s_len - 1] = (s[s_len - 1] - s_mean) / s_std

        for i in range(x_len - s_len + 1):
            # add the last value
            inc_stats_add(
                &stats, 1.0, x[i + s_len - 1]
            )
            std = inc_stats_variance(&stats)
            if std == 0.0:
                std = 1.0
            else:
                std = sqrt(std)

            mean = stats.mean
            for j in range(s_len):
                self.x_buffer[j] = (x[i + j] - mean) / std

            # remove the first value from the mean/std
            inc_stats_remove(&stats, 1.0, x[i])

            if (
                self.wrap._eadistance(
                    self.s_buffer,
                    s_len,
                    self.x_buffer,
                    s_len,
                    &min_dist,
                )
            ):
                if return_index != NULL:
                    return_index[0] = i

        return min_dist

    cdef Py_ssize_t _matches(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        cdef double mean, std
        cdef double tmp_dist
        cdef Py_ssize_t i, j, n

        cdef IncStats stats
        inc_stats_init(&stats)

        for i in range(s_len - 1):
            self.s_buffer[i] = (s[i] - s_mean) / s_std
            inc_stats_add(&stats, 1.0, x[i])

        self.s_buffer[s_len - 1] = (s[s_len - 1] - s_mean) / s_std

        n = 0
        for i in range(x_len - s_len + 1):
            # add the last value
            inc_stats_add(
                &stats, 1.0, x[i + s_len - 1]
            )
            std = inc_stats_variance(&stats)
            if std == 0.0:
                std = 1.0
            else:
                std = sqrt(std)

            mean = stats.mean
            for j in range(s_len):
                self.x_buffer[j] = (x[i + j] - mean) / std

            # remove the first value from the mean/std
            inc_stats_remove(&stats, 1.0, x[i])

            tmp_dist = threshold
            if (
                self.wrap._eadistance(
                    self.s_buffer,
                    s_len,
                    self.x_buffer,
                    s_len,
                    &tmp_dist,
                )
            ):
                if indicies != NULL:
                    indicies[n] = i

                distances[n] = tmp_dist
                n += 1

        return n


cdef class SubsequenceMetricWrap(SubsequenceMetric):
    cdef Metric wrap

    def __cinit__(self, Metric wrap):
        self.wrap = wrap

    def __reduce__(self):
        return self.__class__, (self.wrap, )

    cdef int reset(self, TSArray X) noexcept nogil:
        return self.wrap.reset(X, X)

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        cdef double min_dist = INFINITY
        cdef Py_ssize_t i, j

        for i in range(x_len - s_len + 1):
            if (
                self.wrap._eadistance(
                    s,
                    s_len,
                    x + i,
                    s_len,
                    &min_dist,
                )
            ):
                if return_index != NULL:
                    return_index[0] = i

        return min_dist

    cdef Py_ssize_t _matches(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil:
        cdef double tmp_dist
        cdef Py_ssize_t i, j, n

        n = 0
        for i in range(x_len - s_len + 1):
            tmp_dist = threshold
            if (
                self.wrap._eadistance(
                    s,
                    s_len,
                    x + i,
                    s_len,
                    &tmp_dist,
                )
            ):
                if indicies != NULL:
                    indicies[n] = i

                distances[n] = tmp_dist
                n += 1

        return n

cdef class Metric:

    cdef int reset(self, TSArray x, TSArray y) noexcept nogil:
        return 0

    cdef double distance(
        self,
        TSArray x,
        Py_ssize_t x_index,
        TSArray y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) noexcept nogil:
        return self._distance(
            &x[x_index, dim, 0],
            x.shape[2],
            &y[y_index, dim, 0],
            y.shape[2],
        )

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return NAN

    # Default implementation. Delegates to _eadistance.
    cdef bint eadistance(
        self,
        TSArray x,
        Py_ssize_t x_index,
        TSArray y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
        double *distance,
    ) noexcept nogil:
        return self._eadistance(
           &x[x_index, dim, 0],
           x.shape[2],
           &y[y_index, dim, 0],
           y.shape[2],
           distance,
        )

    # Default implementation. Delegates to _distance,
    # without early abandoning.
    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *min_dist,
    ) noexcept nogil:
        cdef double dist = self._distance(x, x_len, y, y_len)
        if dist < min_dist[0]:
            min_dist[0] = dist
            return True
        else:
            return False

    @property
    def is_elastic(self):
        return False


cdef class CallableMetric(Metric):

    cdef object func

    def __init__(self, func):
        self.func = func

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        cdef np.npy_intp x_shape[1]
        cdef np.npy_intp y_shape[1]
        x_shape[0] = <np.npy_intp> x_len
        y_shape[0] = <np.npy_intp> y_len
        with gil:
            try:
                return float(
                    self.func(
                        np.PyArray_SimpleNewFromData(1, x_shape, np.NPY_DOUBLE, x),
                        np.PyArray_SimpleNewFromData(1, y_shape, np.NPY_DOUBLE, y),
                    )
                )
            except:  # noqa: E722
                return NAN


cdef Py_ssize_t dilated_distance_profile(
    Py_ssize_t stride,
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double *kernel,
    Py_ssize_t k_len,
    const double* x,
    Py_ssize_t x_len,
    Metric metric,
    double *x_buffer,
    double *k_buffer,
    double ea,
    double* out,
) noexcept nogil:
    cdef Py_ssize_t input_size = x_len + 2 * padding
    cdef Py_ssize_t kernel_size = (k_len - 1) * dilation + 1
    cdef Py_ssize_t output_size = <Py_ssize_t> floor((input_size - kernel_size) / stride) + 1

    cdef Py_ssize_t j  # the index in the kernel and input array
    cdef Py_ssize_t i  # the index of the output array
    cdef Py_ssize_t k  # buffer index
    cdef Py_ssize_t padding_offset
    cdef Py_ssize_t input_offset
    cdef Py_ssize_t kernel_offset
    cdef Py_ssize_t convolution_size
    cdef double tmp_dist
    cdef Py_ssize_t dp_size = 0

    for i in range(output_size):
        padding_offset = padding - i * stride
        if padding_offset > 0:
            if padding_offset % dilation == 0:
                kernel_offset = padding_offset
            else:
                kernel_offset = padding_offset + dilation - (padding_offset % dilation)
            input_offset = kernel_offset - padding_offset
        else:
            kernel_offset = 0
            input_offset = labs(padding_offset)

        convolution_size = (
            min(x_len, input_offset + kernel_size - max(0, padding_offset))
            - input_offset
        )
        k = 0
        for j from 0 <= j < convolution_size by dilation:
            x_buffer[k] = x[input_offset + j]
            k_buffer[k] = kernel[((j + kernel_offset) // dilation)]
            k += 1

        tmp_dist = ea
        if metric._eadistance(x_buffer, k, k_buffer, k, &tmp_dist):
            out[dp_size] = tmp_dist / (<float>k / k_len)
            dp_size += 1

    return dp_size


cdef Py_ssize_t scaled_dilated_distance_profile(
    Py_ssize_t stride,
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double *kernel,
    Py_ssize_t k_len,
    const double* x,
    Py_ssize_t x_len,
    Metric metric,
    double *x_buffer,
    double *k_buffer,
    double ea,
    double* out,
) noexcept nogil:
    cdef Py_ssize_t input_size = x_len + 2 * padding
    cdef Py_ssize_t kernel_size = (k_len - 1) * dilation + 1
    cdef Py_ssize_t output_size = <Py_ssize_t> floor((input_size - kernel_size) / stride) + 1

    cdef Py_ssize_t j  # the index in the kernel and input array
    cdef Py_ssize_t i  # the index of the output array
    cdef Py_ssize_t k  # buffer index
    cdef Py_ssize_t padding_offset
    cdef Py_ssize_t input_offset
    cdef Py_ssize_t kernel_offset
    cdef Py_ssize_t convolution_size
    cdef double tmp_dist
    cdef double mean, std
    cdef Py_ssize_t dp_size = 0

    for i in range(output_size):
        padding_offset = padding - i * stride
        if padding_offset > 0:
            if padding_offset % dilation == 0:
                kernel_offset = padding_offset
            else:
                kernel_offset = padding_offset + dilation - (padding_offset % dilation)
            input_offset = kernel_offset - padding_offset
        else:
            kernel_offset = 0
            input_offset = labs(padding_offset)

        convolution_size = (
            min(x_len, input_offset + kernel_size - max(0, padding_offset))
            - input_offset
        )
        k = 0
        mean = 0
        std = 0
        for j from 0 <= j < convolution_size by dilation:
            x_buffer[k] = x[input_offset + j]
            k_buffer[k] = kernel[((j + kernel_offset) // dilation)]
            mean += x_buffer[k]
            std += x_buffer[k] * x_buffer[k]
            k += 1

        mean = mean / k_len
        std = std / k_len - mean * mean
        if std > EPSILON:
            std = sqrt(std)
        else:
            std = 1

        for j in range(k):
            x_buffer[j] = (x_buffer[j] - mean) / std

        tmp_dist = ea
        if metric._eadistance(x_buffer, k, k_buffer, k, &tmp_dist):
            out[dp_size] = tmp_dist / (<float>k / k_len)
            dp_size += 1

    return dp_size


cdef ndarray[intp_t] _new_match_array(Py_ssize_t *matches, Py_ssize_t n_matches):
    if n_matches > 0:
        match_array = np.empty(n_matches, dtype=np.intp)
        for i in range(n_matches):
            match_array[i] = matches[i]
        return match_array
    else:
        return None


cdef ndarray[double_t] _new_distance_array(double *distances, Py_ssize_t n_matches):
    if n_matches > 0:
        dist_array = np.empty(n_matches, dtype=np.double)
        for i in range(n_matches):
            dist_array[i] = distances[i]
        return dist_array
    else:
        return None


cdef class _PairwiseSubsequenceDistance:

    cdef TSArray X
    cdef Py_ssize_t[:, :] min_indices
    cdef double[:, :] distances,
    cdef Subsequence **shapelets
    cdef Py_ssize_t n_shapelets
    cdef SubsequenceMetric metric

    def __cinit__(
        self,
        double[:, :] distances,
        Py_ssize_t[:, :] min_indices,
        TSArray X,
        SubsequenceMetric metric
    ):
        self.X = X
        self.distances = distances
        self.min_indices = min_indices
        self.metric = metric
        self.shapelets = NULL
        self.n_shapelets = 0

    def __dealloc__(self):
        if self.shapelets != NULL:
            for i in range(self.n_shapelets):
                self.metric.free_persistent(self.shapelets[i])
                free(self.shapelets[i])
            free(self.shapelets)
            self.shapelets = NULL

    cdef void set_shapelets(self, list shapelets, Py_ssize_t dim):
        cdef Py_ssize_t i
        self.n_shapelets = len(shapelets)
        self.shapelets = <Subsequence**> malloc(sizeof(Subsequence*) * self.n_shapelets)
        cdef Subsequence *s
        for i in range(self.n_shapelets):
            s = <Subsequence*> malloc(sizeof(Subsequence))
            self.metric.from_array(s, (dim, shapelets[i]))
            self.shapelets[i] = s

    @property
    def n_work(self):
        return self.X.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Py_ssize_t i, j, min_index
        cdef Subsequence *s
        cdef double distance
        cdef SubsequenceMetric metric = deepcopy(self.metric)
        with nogil:
            metric.reset(self.X)
            for i in range(offset, offset + batch_size):
                for j in range(self.n_shapelets):
                    s = self.shapelets[j]
                    distance = metric.persistent_distance(
                        s, self.X, i, &min_index
                    )
                    self.distances[i, j] = distance
                    self.min_indices[i, j] = min_index


def _pairwise_subsequence_distance(
    list shapelets,
    TSArray x,
    int dim,
    SubsequenceMetric metric,
    n_jobs,
):
    n_samples = x.shape[0]
    distances = np.empty((n_samples, len(shapelets)), dtype=np.double)
    min_indicies = np.empty((n_samples, len(shapelets)), dtype=np.intp)
    subsequence_distance = _PairwiseSubsequenceDistance(
        distances,
        min_indicies,
        x,
        metric,
    )
    metric.reset(x)  # TODO: Move to _PairwiseSubsequenceDistance
    subsequence_distance.set_shapelets(shapelets, dim)
    run_in_parallel(subsequence_distance, n_jobs=n_jobs, require="sharedmem")
    return distances, min_indicies


def _paired_subsequence_distance(
    list shapelets,
    TSArray x,
    int dim,
    SubsequenceMetric metric
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef Py_ssize_t i, min_index
    cdef double dist
    cdef Subsequence subsequence
    cdef double[:] distances = np.empty(n_samples, dtype=np.double)
    cdef Py_ssize_t[:] min_indices = np.empty(n_samples, dtype=np.intp)

    metric.reset(x)
    for i in range(n_samples):
        metric.from_array(&subsequence, (dim, shapelets[i]))
        with nogil:
            dist = metric.persistent_distance(&subsequence, x, i, &min_index)
            metric.free_persistent(&subsequence)
            distances[i] = dist
            min_indices[i] = min_index

    return distances.base, min_indices.base


def _subsequence_match(
    object y,
    TSArray x,
    double threshold,
    int dim,
    SubsequenceMetric metric,
    n_jobs,
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef double *distances = <double*> malloc(sizeof(double) * x.shape[2])
    cdef Py_ssize_t *indicies = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * x.shape[2])
    cdef Py_ssize_t i, n_matches

    cdef list distances_list = []
    cdef list indicies_list = []

    cdef Subsequence subsequence
    metric.reset(x)
    metric.from_array(&subsequence, (dim, y))
    with nogil:
        for i in range(x.shape[0]):
            n_matches = metric.persistent_matches(
                &subsequence,
                x,
                i,
                threshold,
                distances,
                indicies,
            )
            with gil:
                indicies_list.append(_new_match_array(indicies, n_matches))
                distances_list.append(_new_distance_array(distances, n_matches))

    free(distances)
    free(indicies)

    metric.free_persistent(&subsequence)
    return indicies_list, distances_list


def _paired_subsequence_match(
    list y,
    TSArray x,
    double threshold,
    int dim,
    SubsequenceMetric metric,
    n_jobs,
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef double *distances = <double*> malloc(sizeof(double) * x.shape[2])
    cdef Py_ssize_t *indices = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * x.shape[2])
    cdef Py_ssize_t i, n_matches

    cdef list distances_list = []
    cdef list indicies_list = []

    cdef Subsequence subsequence
    metric.reset(x)
    for i in range(n_samples):
        metric.from_array(&subsequence, (dim, y[i]))
        n_matches = metric.persistent_matches(
            &subsequence,
            x,
            i,
            threshold,
            distances,
            indices,
        )

        indicies_list.append(_new_match_array(indices, n_matches))
        distances_list.append(_new_distance_array(distances, n_matches))
        metric.free_persistent(&subsequence)

    free(distances)
    free(indices)

    return indicies_list, distances_list


cdef class _PairwiseDistanceBatch:
    cdef TSArray x
    cdef TSArray y
    cdef Py_ssize_t dim
    cdef Metric metric

    cdef double[:, :] distances

    def __init__(
        self,
        TSArray x,
        TSArray y,
        Py_ssize_t dim,
        Metric metric,
        double[:, :] distances
    ):
        self.x = x
        self.y = y
        self.dim = dim
        self.metric = metric
        self.distances = distances

    @property
    def n_work(self):
        return self.x.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Metric metric = deepcopy(self.metric)
        cdef Py_ssize_t i, j
        cdef Py_ssize_t y_samples = self.y.shape[0]

        with nogil:
            metric.reset(self.x, self.y)
            for i in range(offset, offset + batch_size):
                for j in range(y_samples):
                    self.distances[i, j] = metric.distance(
                        self.x, i, self.y, j, self.dim
                    )


def _pairwise_distance(
    TSArray x,
    TSArray y,
    Py_ssize_t dim,
    Metric metric,
    n_jobs,
):

    cdef double[:, :] out = np.empty((x.shape[0], y.shape[0]), dtype=float)
    run_in_parallel(
        _PairwiseDistanceBatch(
            x,
            y,
            dim,
            metric,
            out,
        ),
        n_jobs=n_jobs,
        require="sharedmem",
    )

    return out.base


cdef class _SingletonPairwiseDistanceBatch:
    cdef TSArray x
    cdef Py_ssize_t dim
    cdef Metric metric

    cdef double[:, :] distances

    def __init__(
        self,
        TSArray x,
        Py_ssize_t dim,
        Metric metric,
        double[:, :] distances
    ):
        self.x = x
        self.dim = dim
        self.metric = metric
        self.distances = distances

    @property
    def n_work(self):
        return self.x.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Metric metric = deepcopy(self.metric)
        cdef Py_ssize_t i, j
        cdef Py_ssize_t n_samples = self.x.shape[0]
        cdef double dist

        with nogil:
            metric.reset(self.x, self.x)
            for i in range(offset, offset + batch_size):
                for j in range(i + 1, n_samples):
                    dist = metric.distance(
                        self.x, i, self.x, j, self.dim
                    )
                    self.distances[i, j] = dist
                    self.distances[j, i] = dist


def _singleton_pairwise_distance(
    TSArray x,
    Py_ssize_t dim,
    Metric metric,
    n_jobs
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef double[:, :] out = np.zeros((n_samples, n_samples), dtype=float)

    run_in_parallel(
        _SingletonPairwiseDistanceBatch(
            x,
            dim,
            metric,
            out,
        ),
        n_jobs=n_jobs,
        require="sharedmem",
    )
    return out.base


cdef class _ArgMinBatch:
    cdef double[:, :] distances
    cdef Py_ssize_t[:, :] indices

    cdef TSArray x
    cdef TSArray y
    cdef Metric metric
    cdef Py_ssize_t dim

    def __init__(
        self,
        TSArray x,
        TSArray y,
        Metric metric,
        Py_ssize_t dim,
        double[:, :] distances,
        Py_ssize_t[:, :] indices
    ):
        self.x = x
        self.y = y
        self.metric = metric
        self.dim = dim
        self.distances = distances
        self.indices = indices

    @property
    def n_work(self):
        return self.x.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef double distance
        cdef Py_ssize_t i, j

        cdef Py_ssize_t k = self.distances.shape[1]
        cdef Py_ssize_t y_samples = self.y.shape[0]
        cdef Heap heap = Heap(k)
        cdef HeapElement e
        cdef Metric metric = deepcopy(self.metric)

        with nogil:
            metric.reset(self.x, self.y)
            for i in range(offset, offset + batch_size):
                distance = INFINITY
                heap.reset()

                for j in range(y_samples):
                    if metric.eadistance(self.x, i, self.y, j, self.dim, &distance):
                        heap.push(j, distance)

                        if heap.isfull():
                            distance = heap.maxvalue()
                        else:
                            distance = INFINITY

                for j in range(k):
                    e = heap.getelement(j)
                    self.indices[i, j] = e.index
                    self.distances[i, j] = e.value


def _argmin_distance(
    TSArray x,
    TSArray y,
    Py_ssize_t dim,
    Metric metric,
    Py_ssize_t k,
    n_jobs
):

    cdef Py_ssize_t x_samples = x.shape[0]
    cdef Py_ssize_t y_samples = y.shape[0]

    cdef Py_ssize_t[:, :] indices = np.zeros((x_samples, k), dtype=np.intp)
    cdef double[:, :] values = np.zeros((x_samples, k), dtype=float)

    run_in_parallel(
        _ArgMinBatch(
            x,
            y,
            metric,
            dim,
            values,
            indices,
        ),
        n_jobs=n_jobs,
        require="sharedmem"
    )

    return indices.base, values.base


cdef class _ArgminSubsequenceDistance:

    cdef double[:, :] distances
    cdef Py_ssize_t[:, :] indices

    cdef TSArray s
    cdef Py_ssize_t[:] s_len
    cdef TSArray t
    cdef Metric metric
    cdef Py_ssize_t dim

    def __cinit__(
        self,
        TSArray s,
        Py_ssize_t[:] s_len,
        TSArray t,
        Py_ssize_t dim,
        Metric metric,
        double[:, :] distances,
        Py_ssize_t[:, :] indices,
    ):
        self.s = s
        self.s_len = s_len
        self.t = t
        self.dim = dim
        self.metric = metric
        self.distances = distances
        self.indices = indices

    @property
    def n_work(self):
        return self.t.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef double distance
        cdef Py_ssize_t i, j

        cdef Py_ssize_t k = self.distances.shape[1]
        cdef Heap heap = Heap(k)
        cdef HeapElement e
        cdef Metric metric = deepcopy(self.metric)

        with nogil:
            metric.reset(self.t, self.s)
            for i in range(offset, offset + batch_size):
                distance = INFINITY
                heap.reset()

                for j in range(self.t.shape[2] - self.s_len[i] + 1):
                    if (
                        metric._eadistance(
                            &self.s[i, self.dim, 0],
                            self.s_len[i],
                            &self.t[i, self.dim, j],
                            self.s_len[i],
                            &distance)
                    ):
                        heap.push(j, distance)

                    if heap.isfull():
                        distance = heap.maxvalue()
                    else:
                        distance = INFINITY

                for j in range(k):
                    e = heap.getelement(j)
                    self.indices[i, j] = e.index
                    self.distances[i, j] = e.value


cdef class _ScaledArgminSubsequenceDistance:

    cdef double[:, :] distances
    cdef Py_ssize_t[:, :] indices

    cdef TSArray s
    cdef Py_ssize_t[:] s_len
    cdef TSArray t
    cdef Metric metric
    cdef Py_ssize_t dim

    cdef double *s_buffer
    cdef double *t_buffer

    def __cinit__(
        self,
        TSArray s,
        Py_ssize_t[:] s_len,
        TSArray t,
        Py_ssize_t dim,
        Metric metric,
        double[:, :] distances,
        Py_ssize_t[:, :] indices,
    ):
        self.s = s
        self.s_len = s_len
        self.t = t
        self.dim = dim
        self.metric = metric
        self.distances = distances
        self.indices = indices
        self.s_buffer = <double*> malloc(sizeof(double) * s.shape[2])
        self.t_buffer = <double*> malloc(sizeof(double) * s.shape[2])

    def __dealloc__(self):
        free(self.s_buffer)
        free(self.t_buffer)

    @property
    def n_work(self):
        return self.t.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef double distance
        cdef Py_ssize_t i, j, n
        cdef Py_ssize_t s_len
        cdef Py_ssize_t n_timestep = self.t.shape[2]
        cdef Py_ssize_t k = self.distances.shape[1]

        cdef Heap heap = Heap(k)
        cdef HeapElement e
        cdef Metric metric = deepcopy(self.metric)

        cdef double mean, std
        cdef IncStats stats
        with nogil:
            metric.reset(self.t, self.s)
            for i in range(offset, offset + batch_size):
                s_len = self.s_len[i]

                distance = INFINITY
                heap.reset()
                inc_stats_init(&stats)

                fast_mean_std(&self.s[i, self.dim, 0], s_len, &mean, &std)
                if std == 0.0:
                    std = 1.0

                for n in range(s_len - 1):
                    self.s_buffer[n] = (self.s[i, self.dim, n] - mean) / std
                    inc_stats_add(&stats, 1.0, self.t[i, self.dim, n])

                self.s_buffer[s_len - 1] = (
                    self.s[i, self.dim, s_len - 1] - mean
                ) / std

                for j in range(n_timestep - s_len + 1):
                    # add the last value
                    inc_stats_add(
                        &stats, 1.0, self.t[i, self.dim, j + s_len - 1]
                    )
                    std = inc_stats_variance(&stats)
                    if std == 0.0:
                        std = 1.0
                    else:
                        std = sqrt(std)

                    mean = stats.mean
                    for n in range(s_len):
                        self.t_buffer[n] = (self.t[i, self.dim, j + n] - mean) / std

                    # remove the first value from the mean/std
                    inc_stats_remove(&stats, 1.0, self.t[i, self.dim, j])

                    if (
                        metric._eadistance(
                            self.s_buffer,
                            s_len,
                            self.t_buffer,
                            s_len,
                            &distance)
                    ):
                        heap.push(j, distance)

                    if heap.isfull():
                        distance = heap.maxvalue()
                    else:
                        distance = INFINITY

                for j in range(k):
                    e = heap.getelement(j)
                    self.indices[i, j] = e.index
                    self.distances[i, j] = e.value

def _argmin_subsequence_distance(
    TSArray s,
    Py_ssize_t[:] s_len,
    TSArray x,
    Py_ssize_t dim,
    Metric metric,
    Py_ssize_t k,
    bint scaled=False,
    n_jobs=None,
):
    cdef Py_ssize_t n_samples = x.shape[0]
    assert s.shape[0] == n_samples

    cdef Py_ssize_t[:, :] indices = np.empty((n_samples, k), dtype=np.intp)
    cdef double[:, :] values = np.empty((n_samples, k), dtype=float)

    if scaled:
        Method = _ScaledArgminSubsequenceDistance
    else:
        Method = _ArgminSubsequenceDistance

    run_in_parallel(
        Method(
            s,
            s_len,
            x,
            dim,
            metric,
            values,
            indices,
        ),
        n_jobs=n_jobs,
        require="sharedmem",
    )

    return indices.base, values.base


cdef class _PairedDistanceBatch:
    cdef TSArray x
    cdef TSArray y
    cdef Py_ssize_t dim
    cdef Metric metric

    cdef double[:] distances

    def __init__(
        self,
        TSArray x,
        TSArray y,
        Py_ssize_t dim,
        Metric metric,
        double[:] distances,
    ):
        self.x = x
        self.y = y
        self.dim = dim
        self.metric = metric
        self.distances = distances

    @property
    def n_work(self):
        return self.x.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Metric metric = deepcopy(self.metric)
        cdef Py_ssize_t i
        with nogil:
            metric.reset(self.x, self.y)
            for i in range(offset, offset + batch_size):
                self.distances[i] = metric.distance(self.x, i, self.y, i, self.dim)


def _paired_distance(
    TSArray y,
    TSArray x,
    Py_ssize_t dim,
    Metric metric,
    n_jobs,
):
    cdef double[:] out = np.empty(y.shape[0], dtype=float)
    run_in_parallel(
        _PairedDistanceBatch(
            x,
            y,
            dim,
            metric,
            out,
        ),
        n_jobs=n_jobs,
        require="sharedmem",
    )

    return out.base


cdef class _DistanceProfile:
    cdef double[:, :] distance_profile
    cdef TSArray x
    cdef TSArray y
    cdef Py_ssize_t dim
    cdef SubsequenceMetric metric

    def __cinit__(
        self,
        TSArray y,
        TSArray x,
        Py_ssize_t dim,
        SubsequenceMetric metric,
        double[:, :] distance_profile,
    ):
        self.distance_profile = distance_profile
        self.x = x
        self.y = y
        self.metric = metric
        self.dim = dim

    @property
    def n_work(self):
        return self.x.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Py_ssize_t i, j
        cdef SubsequenceMetric metric = deepcopy(self.metric)
        cdef SubsequenceView view

        with nogil:
            metric.reset(self.x)
            for i in range(offset, offset + batch_size):
                metric.init_transient(self.y, &view, i, 0, self.y.shape[2], 0)
                metric._distance_profile(
                    &self.y[i, 0, 0],
                    self.y.shape[2],
                    view.mean,
                    view.std,
                    view.extra,
                    &self.x[i, 0, 0],
                    self.x.shape[2],
                    &self.distance_profile[i, 0],
                )
                metric.free_transient(&view)


def _distance_profile(
    TSArray y,
    TSArray x,
    Py_ssize_t dim,
    SubsequenceMetric metric,
    n_jobs=None
):
    cdef double[:, :] out = np.ones(
        (x.shape[0], x.shape[2] - y.shape[2] + 1), dtype=float
    )

    run_in_parallel(
        _DistanceProfile(
            y,
            x,
            dim,
            metric,
            out,
        ),
        n_jobs=n_jobs,
        require="sharedmem",
    )
    return out.base


cdef class _DilatedDistanceProfile:
    cdef TSArray S
    cdef TSArray X
    cdef Py_ssize_t dim
    cdef Metric metric
    cdef Py_ssize_t dilation
    cdef Py_ssize_t padding
    cdef bint scaled,
    cdef double[:, :] out
    cdef double *x_buffer
    cdef double *s_buffer

    def __cinit__(
        self,
        TSArray S,
        TSArray X,
        Py_ssize_t shapelet_size,
        Py_ssize_t dim,
        Metric metric,
        Py_ssize_t dilation,
        Py_ssize_t padding,
        bint scaled,
        double[:, :] out,
    ):
        self.S = S
        self.X = X
        self.dim = dim
        self.metric = metric
        self.dilation = dilation
        self.padding = padding
        self.scaled = scaled
        self.out = out
        self.x_buffer = <double*> malloc(sizeof(double) * shapelet_size)
        self.s_buffer = <double*> malloc(sizeof(double) * shapelet_size)

    def __dealloc__(self):
        free(self.x_buffer)
        free(self.s_buffer)

    @property
    def n_work(self):
        return self.X.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Metric metric = deepcopy(self.metric)
        cdef Py_ssize_t i

        with nogil:
            metric.reset(self.X, self.X)
            if self.scaled:
                for i in range(offset, offset + batch_size):
                    scaled_dilated_distance_profile(
                        1,
                        self.dilation,
                        self.padding,
                        &self.S[i, self.dim, 0],
                        self.S.shape[2],
                        &self.X[i, self.dim, 0],
                        self.X.shape[2],
                        metric,
                        self.x_buffer,
                        self.s_buffer,
                        INFINITY,
                        &self.out[i, 0],
                    )
            else:
                for i in range(offset, offset + batch_size):
                    dilated_distance_profile(
                        1,
                        self.dilation,
                        self.padding,
                        &self.S[i, self.dim, 0],
                        self.S.shape[2],
                        &self.X[i, self.dim, 0],
                        self.X.shape[2],
                        metric,
                        self.x_buffer,
                        self.s_buffer,
                        INFINITY,
                        &self.out[i, 0],
                    )


def _dilated_distance_profile(
    TSArray S,
    TSArray X,
    Py_ssize_t dim,
    Metric metric,
    Py_ssize_t dilation,
    Py_ssize_t padding,
    bint scaled,
    n_jobs,
):
    cdef Py_ssize_t shapelet_size = (S.shape[2] - 1) * dilation + 1
    cdef Py_ssize_t input_size = X.shape[2] + 2 * padding
    cdef double[:, :] out = np.empty(
        (X.shape[0], input_size - shapelet_size + 1), dtype=float
    )

    run_in_parallel(
        _DilatedDistanceProfile(
            S,
            X,
            shapelet_size,
            dim,
            metric,
            dilation,
            padding,
            scaled,
            out,
        ),
        n_jobs=n_jobs,
        require="sharedmem",
    )

    return out.base
