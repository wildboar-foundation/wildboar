# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

cimport numpy as np
from libc.math cimport NAN, sqrt
from libc.stdlib cimport free, malloc

from ..utils cimport stats

from copy import deepcopy


from ..utils.data cimport Dataset

from ..utils.data import check_dataset
from ..utils.parallel import run_in_parallel
from ..utils.validation import check_array


cdef double EPSILON = 1e-13

cdef int _ts_view_update_statistics(SubsequenceView *v, Dataset td) nogil:
    """Update the mean and standard deviation of a shapelet info struct """
    cdef Py_ssize_t shapelet_offset = (
            v.index * td.sample_stride +
            v.dim * td.dim_stride +
            v.start
    )
    cdef double ex = 0
    cdef double ex2 = 0
    cdef Py_ssize_t i
    for i in range(v.length):
        current_value = td.data[shapelet_offset + i]
        ex += current_value
        ex2 += current_value ** 2

    v.mean = ex / v.length
    ex2 = ex2 / v.length - v.mean * v.mean
    if ex2 > EPSILON:
        v.std = sqrt(ex2)
    else:
        v.std = 0.0
    return 0


cdef class SubsequenceDistanceMeasure:

    cdef int reset(self, Dataset dataset) nogil:
        pass

    cdef int init_transient(
        self,
        Dataset td,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil:
        v.index = index
        v.dim = dim
        v.start = start
        v.length = length
        v.mean = NAN
        v.std = NAN
        v.extra = NULL
        return 0

    cdef int init_persistent(
        self,
        Dataset dataset,
        SubsequenceView* v,
        Subsequence* s,
    ) nogil:
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

        cdef double *sample = dataset.get_sample(v.index, v.dim) + v.start
        cdef Py_ssize_t i
        for i in range(v.length):
            s.data[i] = sample[i]
        return 0

    cdef int free_transient(self, SubsequenceView *v) nogil:
        return 0

    cdef int free_persistent(self, Subsequence *v) nogil:
        if v.data != NULL:
            free(v.data)
            v.data = NULL
        return 0

    cdef int from_array(
        self,
        Subsequence *v,
        object obj,
    ):
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

    cdef object to_array(
        self, 
        Subsequence *s
    ):
        cdef Py_ssize_t j
        arr = np.empty(s.length, dtype=float)
        for j in range(s.length):
            arr[j] = s.data[j]

        return (s.dim, arr)

    cdef double transient_distance(
        self,
        SubsequenceView *s,
        Dataset dataset,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        return self._distance(
            dataset.get_sample(s.index, s.dim) + s.start,
            s.length,
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            return_index,
        )

    cdef double persistent_distance(
        self,
        Subsequence *s,
        Dataset dataset,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        return self._distance(
            s.data,
            s.length,
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            return_index,
        )

    cdef Py_ssize_t transient_matches(
        self,
        SubsequenceView *v,
        Dataset dataset,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil:
        return self._matches(
            dataset.get_sample(v.index, v.dim) + v.start,
            v.length,
            dataset.get_sample(index, v.dim),
            dataset.n_timestep,
            threshold,
            distances,
            indicies,
        )

    cdef Py_ssize_t persistent_matches(
        self,
        Subsequence *s,
        Dataset dataset,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil:
        return self._matches(
            s.data,
            s.length,
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            threshold,
            distances,
            indicies,
        )

    cdef double _distance(
        self,
        double *s,
        Py_ssize_t s_len,
        double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        return NAN

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil:
        return -1


cdef class ScaledSubsequenceDistanceMeasure(SubsequenceDistanceMeasure):

    cdef int init_transient(
        self,
        Dataset dataset,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil:
        cdef int err = SubsequenceDistanceMeasure.init_transient(
            self, dataset, v, index, start, length, dim
        )
        if err < 0:
            return err
        _ts_view_update_statistics(v, dataset)

    cdef int from_array(
        self,
        Subsequence *v,
        object obj,
    ):
        cdef int err = SubsequenceDistanceMeasure.from_array(self, v, obj)
        if err == -1:
            return -1

        dim, arr = obj
        v.mean = np.mean(arr)
        v.std = np.std(arr)
        if v.std <= EPSILON:
            v.std = 0.0
        return 0


cdef class DistanceMeasure:

    cdef int reset(self, Dataset x, Dataset y) nogil:
        return 0

    cdef double distance(
        self,
        Dataset x,
        Py_ssize_t x_index,
        Dataset y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) nogil:
        return self._distance(
            x.get_sample(x_index, dim),
            x.n_timestep,
            y.get_sample(y_index, dim),
            y.n_timestep,
        )

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len
    ) nogil:
        return NAN

    @property
    def is_elastic(self):
        return False


cdef class FuncDistanceMeasure:
    cdef object func
    cdef np.ndarray x_buffer
    cdef np.ndarray y_buffer
    cdef bint _support_unaligned

    def __cinit__(self, Py_ssize_t n_timestep, object func, bint support_unaligned=False):
        self.n_timestep = n_timestep
        self.func = func
        self.x_buffer = np.empty(n_timestep, dtype=float)
        self.y_buffer = np.empty(n_timestep, dtype=float)
        self._support_unaligned = support_unaligned


    def __reduce__(self):
        return self.__class__, (self.n_timestep, self.func)


    cdef double ts_copy_sub_distance(
        self,
        Subsequence *s,
        Dataset td,
        Py_ssize_t t_index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        cdef Py_ssize_t i
        cdef Py_ssize_t sample_offset = (
            t_index * td.sample_stride + 
            s.dim * td.dim_stride
        )
        with gil:
            for i in range(td.n_timestep):
                if i < s.length:
                    self.x_buffer[i] = s.data[i]
                self.y_buffer[i] = td.data[sample_offset + i]

            return self.func(self.x_buffer[:s.length], self.y_buffer)

    cdef double ts_view_sub_distance(
        self,
        SubsequenceView *v,
        Dataset td,
        Py_ssize_t t_index,
    ) nogil:
        cdef Py_ssize_t i
        cdef Py_ssize_t sample_offset = (t_index * td.sample_stride +
                                         v.dim * td.dim_stride)
        cdef Py_ssize_t shapelet_offset = (v.index * td.sample_stride +
                                           v.dim * td.dim_stride +
                                           v.start)
        with gil:
            for i in range(td.n_timestep):
                if i < v.length:
                    self.x_buffer[i] = td.data[shapelet_offset + i]
                self.y_buffer[i] = td.data[sample_offset + i]

            return self.func(self.x_buffer[:v.length], self.y_buffer)

    cdef double ts_copy_distance(
        self,
        Subsequence *s,
        Dataset td,
        Py_ssize_t t_index,
    ) nogil:
        return self.ts_copy_sub_distance(s, td, t_index)

    cdef bint support_unaligned(self) nogil:
        return self._support_unaligned



cdef np.ndarray _new_match_array(Py_ssize_t *matches, Py_ssize_t n_matches):
    if n_matches > 0:
        match_array = np.empty(n_matches, dtype=np.intp)
        for i in range(n_matches):
            match_array[i] = matches[i]
        return match_array
    else:
        return None


cdef np.ndarray _new_distance_array(
        double *distances, Py_ssize_t n_matches):
    if n_matches > 0:
        dist_array = np.empty(n_matches, dtype=np.double)
        for i in range(n_matches):
            dist_array[i] = distances[i]
        return dist_array
    else:
        return None


cdef class _PairwiseSubsequenceDistance:

    cdef Dataset dataset
    cdef Py_ssize_t[:, :] min_indices
    cdef double[:, :] distances,
    cdef Subsequence **shapelets
    cdef Py_ssize_t n_shapelets
    cdef SubsequenceDistanceMeasure distance_measure

    def __cinit__(
        self, 
        double[:, :] distances,
        Py_ssize_t[:, :] min_indices,
        Dataset dataset, 
        SubsequenceDistanceMeasure distance_measure
    ):
        self.dataset = dataset
        self.distances = distances
        self.min_indices = min_indices
        self.distance_measure = distance_measure
        self.shapelets = NULL
        self.n_shapelets = 0

    def __dealloc__(self):
        if self.shapelets != NULL:
            for i in range(self.n_shapelets):
                self.distance_measure.free_persistent(self.shapelets[i])
                free(self.shapelets[i])
            free(self.shapelets)
            self.shapelets = NULL

    cdef void set_shapelets(self, list shapelets, Py_ssize_t dim):
        cdef Py_ssize_t i
        self.n_shapelets = len(shapelets)
        self.shapelets = <Subsequence**> malloc(sizeof(Subsequence*) * self. n_shapelets)
        cdef Subsequence *s
        for i in range(self.n_shapelets):
            s = <Subsequence*> malloc(sizeof(Subsequence))
            self.distance_measure.from_array(s, (dim, shapelets[i]))
            self.shapelets[i] = s

    @property
    def n_work(self):
        return self.dataset.n_samples

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Py_ssize_t i, j, min_index
        cdef Subsequence *s
        cdef double distance
        cdef SubsequenceDistanceMeasure distance_measure = deepcopy(self.distance_measure)
        with nogil:
            distance_measure.reset(self.dataset)
            for i in range(offset, offset + batch_size):
                for j in range(self.n_shapelets):
                    s = self.shapelets[j]
                    distance = distance_measure.persistent_distance(
                        s, self.dataset, i, &min_index
                    )
                    self.distances[i, j] = distance
                    self.min_indices[i, j] = min_index


def _pairwise_subsequence_distance(
    list shapelets, 
    np.ndarray x, 
    int dim, 
    SubsequenceDistanceMeasure distance_measure,
    n_jobs,
):
    x = check_dataset(x)
    dataset = Dataset(x)
    distances = np.empty((dataset.n_samples, len(shapelets)), dtype=np.double)
    min_indicies = np.empty((dataset.n_samples, len(shapelets)), dtype=np.intp)
    subsequence_distance = _PairwiseSubsequenceDistance(
        distances,
        min_indicies,
        dataset,
        distance_measure,
    )
    distance_measure.reset(dataset)
    subsequence_distance.set_shapelets(shapelets, dim)
    run_in_parallel(subsequence_distance, n_jobs=n_jobs, require="sharedmem")
    return distances, min_indicies


def _paired_subsequence_distance(
    list shapelets,
    np.ndarray x,
    int dim,
    SubsequenceDistanceMeasure distance_measure
):
    x = check_dataset(x)
    cdef Dataset dataset = Dataset(x)
    cdef np.ndarray distances = np.empty(dataset.n_samples, dtype=np.double)
    cdef np.ndarray min_indicies = np.empty(dataset.n_samples, dtype=np.intp)
    cdef Py_ssize_t i, min_index
    cdef double dist
    cdef Subsequence subsequence
    distance_measure.reset(dataset)
    for i in range(dataset.n_samples):
        distance_measure.from_array(&subsequence, (dim, shapelets[i]))
        with nogil:
            dist = distance_measure.persistent_distance(&subsequence, dataset, i, &min_index)
        distance_measure.free_persistent(&subsequence)
        distances[i] = dist
        min_indicies[i] = min_index

    return distances, min_indicies


def _subsequence_match(
    np.ndarray y,
    np.ndarray x,
    double threshold,
    int dim,
    SubsequenceDistanceMeasure distance_measure,
    n_jobs,
):
    x = check_dataset(x)
    cdef Dataset dataset = Dataset(x)
    cdef double *distances 
    cdef Py_ssize_t *indicies
    cdef Py_ssize_t i, n_matches

    cdef list distances_list = []
    cdef list indicies_list = []

    cdef Subsequence subsequence
    distance_measure.reset(dataset)
    distance_measure.from_array(&subsequence, (dim, y))
    for i in range(dataset.n_samples):
        n_matches = distance_measure.persistent_matches(
            &subsequence,
            dataset,
            i,
            threshold,
            &distances,
            &indicies,
        )
        indicies_list.append(_new_match_array(indicies, n_matches))
        distances_list.append(_new_distance_array(distances, n_matches))
        free(distances)
        free(indicies)
    distance_measure.free_persistent(&subsequence)
    return indicies_list, distances_list


def _paired_subsequence_match(
    list y,
    np.ndarray x,
    double threshold,
    int dim,
    SubsequenceDistanceMeasure distance_measure,
    n_jobs,
):
    x = check_dataset(x)
    cdef Dataset dataset = Dataset(x)
    cdef double *distances 
    cdef Py_ssize_t *indicies
    cdef Py_ssize_t i, n_matches

    cdef list distances_list = []
    cdef list indicies_list = []

    cdef Subsequence subsequence
    distance_measure.reset(dataset)
    for i in range(dataset.n_samples):
        distance_measure.from_array(&subsequence, (dim, y[i]))
        n_matches = distance_measure.persistent_matches(
            &subsequence,
            dataset,
            i,
            threshold,
            &distances,
            &indicies,
        )
        indicies_list.append(_new_match_array(indicies, n_matches))
        distances_list.append(_new_distance_array(distances, n_matches))
        free(distances)
        free(indicies)
        distance_measure.free_persistent(&subsequence)

    return indicies_list, distances_list


def _pairwise_distance(
    np.ndarray y,
    np.ndarray x,
    Py_ssize_t dim,
    DistanceMeasure distance_measure,
    n_jobs,
):
    y = check_dataset(y, allow_1d=True)
    x = check_dataset(x, allow_1d=True)
    cdef Dataset y_dataset = Dataset(y)
    cdef Dataset x_dataset = Dataset(x)
    cdef np.ndarray out = np.empty((y_dataset.n_samples, x_dataset.n_samples), dtype=np.double)
    cdef double[:, :] out_view = out
    cdef Py_ssize_t i, j
    cdef double dist

    with nogil:
        distance_measure.reset(y_dataset, x_dataset)
        for i in range(y_dataset.n_samples):
            for j in range(x_dataset.n_samples):
                dist = distance_measure.distance(y_dataset, i, x_dataset, j, dim)
                out_view[i, j] = dist

    return out


def _singleton_pairwise_distance(
    np.ndarray x, 
    Py_ssize_t dim, 
    DistanceMeasure distance_measure, 
    n_jobs
):
    x = check_dataset(x)
    cdef Dataset dataset = Dataset(x)
    cdef np.ndarray out = np.zeros((dataset.n_samples, dataset.n_samples), dtype=np.double)
    cdef double[:, :] out_view = out
    cdef Py_ssize_t i, j
    cdef double dist

    with nogil:
        distance_measure.reset(dataset, dataset)
        for i in range(dataset.n_samples):
            for j in range(i + 1, dataset.n_samples):
                dist = distance_measure.distance(dataset, i, dataset, j, dim)
                out_view[i, j] = dist
                out_view[j, i] = dist

    return out


def _paired_distance(
    np.ndarray y,
    np.ndarray x,
    Py_ssize_t dim,
    DistanceMeasure distance_measure,
    n_jobs,
):
    y = check_dataset(y, allow_1d=True)
    x = check_dataset(x, allow_1d=True)
    cdef Dataset y_dataset = Dataset(y)
    cdef Dataset x_dataset = Dataset(x)
    cdef np.ndarray out = np.empty(y_dataset.n_samples, dtype=np.double)
    cdef double[:] out_view = out
    cdef Py_ssize_t i

    with nogil:
        distance_measure.reset(y_dataset, x_dataset)
        for i in range(y_dataset.n_samples):
            dist = distance_measure.distance(y_dataset, i, x_dataset, i, dim)
            out_view[i] = dist

    return out
