from __future__ import print_function

import numpy as np

cimport numpy as np
cimport cython

from libc.math cimport fabs, log2, INFINITY
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

from numpy.random import RandomState

from pypf._distribution cimport get_class_distribution
from pypf._sliding_distance cimport (
    sliding_distance,
    SlidingDistance,
    ShapeletInfo,
    shapelet_info_update_statistics,
    shapelet_info_distances,
    shapelet_info_extract_shapelet,
    Shapelet,
    new_sliding_distance,
    free_sliding_distance    
)

from pypf._impurity cimport safe_info, info

from pypf._utils cimport (
    label_distribution,
    intp_ndarray_to_size_t_ptr,
    argsort,
    rand_int,
    RAND_R_MAX
)
    


import numpy as np
cimport numpy as np
cimport cython
import time

def test(X, i):
    cdef np.ndarray[np.intp_t] np_sample = np.arange(10)
    cdef np.ndarray[np.float64_t] np_values = np.random.randn(10)
    cdef size_t* samples = <size_t*> np_sample.data 
    cdef double* values = <double*> np_values.data

    cdef size_t rand_state = 123
    cdef np.ndarray[np.intp_t] fuck = np.zeros(20, dtype=np.intp)
    for p in range(10000):
        fuck[rand_int(0, 20, &rand_state)] += 1

    print(fuck)
    print(np_sample)
    print(np_values)
    argsort(values, samples, 10)
    for p in range(10):
        print(samples[p], values[p])

    cdef size_t* left = <size_t*> malloc(sizeof(size_t) * 4)
    cdef size_t* right = <size_t*> malloc(sizeof(size_t) * 6)

    memcpy(left, samples, sizeof(size_t) * 4)
    memcpy(right, samples + 4, sizeof(size_t) * 6)

    for p in range(4):
        print(left[p])

    for p in range(6):
        print(right[p])

    
    cdef SlidingDistance sd = new_sliding_distance(X)
    cdef ShapeletInfo s
    s.index = 0
    s.start = 0
    s.length = 100
    shapelet_info_update_statistics(&s, sd)
    print(s.mean, s.std)

    cdef Shapelet shapelet = shapelet_info_extract_shapelet(s, sd)
    print(shapelet.length)
    for p in range(10):
        print(shapelet.data[p])
    
    cdef size_t* indicies = <size_t*> malloc(sizeof(size_t) * i.shape[0])
    intp_ndarray_to_size_t_ptr(i, indicies)
    cdef np.ndarray[np.float64_t] result = np.empty(X.shape[0])
    c = time.time()
    shapelet_info_distances(s, indicies, i.shape[0], sd, <double*>result.data)
    print((time.time() - c) * 1000)

    for i in range(10):
        print(i, result[i])
    free(indicies)
    free_sliding_distance(sd)

cdef class ShapeletTreeBuilder:
    cdef size_t random_seed
    cdef size_t n_shapelets
    
    cdef size_t* labels
    cdef size_t label_stride
    cdef size_t n_labels

    cdef size_t* sample_buffer
    cdef double* distance_buffer
    cdef double* left_label_buffer
    cdef double* right_label_buffer
    
    cdef SlidingDistance sd
    
   
    def __cinit__(self, size_t n_shapelets, object random_state):
        self.random_seed = random_state.randint(0, RAND_R_MAX)
        self.n_shapelets = n_shapelets
        
    def __dealloc__(self):
        self._free_if_needed()

    cdef void _free_if_needed(self):
        if self.sd.X_buffer != NULL:
            free_sliding_distance(self.sd)
            
        if self.labels != NULL:
            free(self.labels)
            self.labels = NULL

        if self.sample_buffer != NULL:
            free(self.sample_buffer)
            self.sample_buffer = NULL

        if self.distance_buffer != NULL:
            free(self.distance_buffer)
            self.distance_buffer = NULL

        if self.left_label_buffer != NULL:
            free(self.left_label_buffer)
            self.left_label_buffer = NULL

        if self.right_label_buffer != NULL:
            free(self.right_label_buffer)
            self.right_label_buffer = NULL
        

    cdef void init(self,
                   np.ndarray[np.float64_t, ndim=2, mode="c"] X,
                   np.ndarray[np.intp_t, ndim=1, mode="c"] y,
                   size_t n_labels):
        self._free_if_needed()

        self.sd = new_sliding_distance(X)
        self.labels = <size_t*> y.data
        self.label_stride = <size_t> y.strides[0] / <size_t> y.itemsize
        self.n_labels = n_labels
        self.left_label_buffer = <double*> malloc(sizeof(double) * n_labels)
        self.right_label_buffer= <double*> malloc(sizeof(double) * n_labels)

    cdef object build_tree(self, np.ndarray[np.intp_t, ndim=1, mode="c"] indicies):
        # indicies must have stride = 1
        cdef const size_t* samples = <size_t*>indicies.data
        cdef size_t n_samples = indicies.shape[0]

        if self.sample_buffer != NULL:
            free(self.sample_buffer)

        self.sample_buffer = <size_t*> malloc(sizeof(size_t) * n_samples)
        self.distance_buffer = <double*> malloc(sizeof(double) * n_samples)
        
        self._build_tree(samples, n_samples)
        # samples are free

    cdef int _build_tree(self, const size_t* samples, size_t n_samples):
        cdef double* dist = <double*> malloc(sizeof(double) * self.n_labels)
        cdef int n_positive = label_distribution(samples, n_samples,
                                                 self.labels,
                                                 self.n_labels, dist)
        if n_samples < 2 or n_positive < 2:
            return -1 # TODO: return leaf

        cdef size_t* new_samples = <size_t*> malloc(sizeof(size_t) * n_samples)
        cdef SplitPoint split = self._split(samples, n_samples, new_samples)

        # samples are now ordered
        
        cdef size_t left_size = split.split_point
        cdef size_t right_size = n_samples - split.split_point
        cdef size_t* left_samples
        cdef size_t* right_samples
        if left_size > 0 and right_size > 0:
            left_samples = <size_t*> malloc(sizeof(size_t) * left_size)
            memcpy(left_samples, samples, sizeof(size_t) * left_size)

            right_samples = <size_t*> malloc(sizeof(size_t) * (n_samples - left_size))
            memcpy(right_samples, samples + left_size, sizeof(size_t) * right_size)

            free(new_samples)
            
            self._build_tree(left_samples, left_size)
            free(left_samples)

            self._build_tree(right_samples, right_size)
            free(right_samples)

            return 2 # return a branch
        else:
            return -1 # TODO: return leaf

    cdef SplitPoint _split(self, const size_t* samples, size_t n_samples, size_t* new_samples):
        cdef double best_impurity = INFINITY
        cdef SplitPoint best_split_point

        cdef size_t split_point
        cdef double threshold
        cdef double impurity
        cdef ShapeletInfo current_shapelet
        
        cdef size_t i
        for i in range(self.n_shapelets):
            memcpy(self.sample_buffer, samples, sizeof(size_t) * n_samples)

            current_shapelet = self._sample_shapelet(samples, n_samples)
            shapelet_info_distances(current_shapelet, samples,
                                    n_samples, self.sd, self.distance_buffer)
            
            argsort(self.distance_buffer, self.sample_buffer, n_samples)
            self._partition_distance_buffer(n_samples, &split_point, &threshold, &impurity)

            if impurity < best_impurity:
                best_split_point.split_point = split_point
                best_split_point.threshold = threshold
                best_split_point.shapelet_info = current_shapelet
                memcpy(new_samples, self.sample_buffer, sizeof(size_t) * n_samples)

        return best_split_point

    cdef ShapeletInfo _sample_shapelet(self, const size_t* samples, size_t n_samples) nogil:
        cdef ShapeletInfo shapelet_info
        
        shapelet_info.length = rand_int(3, self.sd.n_timestep, &self.random_seed)
        shapelet_info.start = rand_int(0, self.sd.n_timestep - shapelet_info.length, &self.random_seed)
        shapelet_info.index = samples[rand_int(0, n_samples, &self.random_seed)]
        
        shapelet_info_update_statistics(&shapelet_info, self.sd)
        return shapelet_info

    cdef void _partition_distance_buffer(self,
                                         size_t n_samples,
                                         size_t* split_point,
                                         double* threshold,
                                         double* impurity):
        memset(self.left_label_buffer, 0, sizeof(double) * self.n_labels)
        
        cdef size_t i
        for i in range(self.n_samples):
            self.right_label_buffer[self.labels[self.sample_buffer[i] * self.label_stride]] = 1.0

        cdef double prev_dist = self.distance_buffer[0]
        cdef size_t prev_label = self.label[self.sample_buffer[0] * self.label_stride]

        cdef double left_sum = 1
        cdef double right_sum = n_samples - 1

        self.left_label_buffer[prev_label] += 1
        self.right_label_buffer[prev_label] -= 1

        impurity[0] = info(left_sum,
                           self.left_label_buffer,
                           right_sum,
                           self.right_label_buffer,
                           n_samples,
                           self.n_labels)
        threshold[0] = prev_dist
        split_point[0] = 1

        cdef size_t j # order index
        
        cdef double dist
        cdef double tmp_impurity
        cdef size_t current_label
        
        for i in range(1, n_samples):
            j = self.sample_buffer[i]
            dist = self.distance_buffer[i]
            label = self.labels[j * self.sample_stride]

            left_sum += 1
            right_sum -= 1
            self.left_label_buffer[label] += 1
            self.right_label_buffer[label] -= 1
            
            if not label == prev_label:
                tmp_impurity = info(left_sum,
                                    self.left_label_buffer,
                                    right_sum,
                                    self.right_label_buffer,
                                    n_samples,
                                    self.n_labels)
                
                if tmp_impurity < impurity[0]:
                    impurity[0] = tmp_impurity
                    threshold[0] = (dist + prev_dist) / 2
                    split_point[0] = i
        
                       

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int partition(long[:] idx,
                   const Py_ssize_t[:] order,
                   double[:] d,
                   long[:] y,
                   Py_ssize_t n_classes,
                   double[:] left_d,
                   double[:] right_d,
                   Py_ssize_t* threshold_index,
                   double* threshold,
                   double* entropy) nogil:
    cdef Py_ssize_t n_examples = idx.shape[0]
    cdef Py_ssize_t example
    cdef Py_ssize_t cls
    for example in range(n_examples):
        right_d[y[idx[example]]] += 1.0
        
    for cls in range(n_classes):
        left_d[cls] = 0.0

    cdef double prev_dist = d[order[0]]
    cdef Py_ssize_t prev_label = y[idx[order[0]]]

    cdef double lt_w = 1
    cdef double gt_w = n_examples - 1
    left_d[prev_label] += 1
    right_d[prev_label] -= 1

    entropy[0] = safe_info(lt_w, left_d, gt_w, right_d, n_examples)
    threshold[0] = prev_dist
    threshold_index[0] = 1

    cdef Py_ssize_t i, order_i
    cdef double dist, e
    cdef Py_ssize_t label
    for i in range(1, n_examples):
        order_i = order[i]
        dist = d[order_i]
        label = y[idx[order_i]]
        if not label == prev_label:
            e = safe_info(lt_w, left_d, gt_w, right_d, n_examples)
            if e < entropy[0]:
#                print(e, lt_w, np.asarray(left_d), gt_w, np.asarray(right_d), n_examples)
                entropy[0] = e
                threshold[0] = (dist + prev_dist) / 2
                threshold_index[0] = i

        prev_label = label
        prev_dist = dist

        lt_w += 1
        gt_w -= 1
        left_d[label] += 1
        right_d[label] -= 1
    return 0

cdef double[:] draw_shapelet(long[:] idx, double[:, :] x, object random_state):
    cdef Py_ssize_t length = random_state.randint(3, x.shape[1])
    cdef Py_ssize_t start = random_state.randint(0, x.shape[1] - length)
    cdef Py_ssize_t i = random_state.randint(len(idx))
    cdef double[:] shapelet = x[idx[i], start:(start + length)]
    cdef double std = np.std(shapelet)
    if std > 0:
        shapelet = (shapelet - np.mean(shapelet)) / std
        return shapelet
    else:
        return np.zeros(length)


cdef class Split:
    cdef long[:] order
    cdef Py_ssize_t threshold_index
    cdef double threshold
    cdef double[:] shapelet    

cdef Split find_threshold(long[:] idx,
                          double[:, :] x,
                          long[:] y,
                          Py_ssize_t n_classes,
                          int n_shapelets,
                          random_state):
    cdef double best_impurity = INFINITY
    cdef np.ndarray[Py_ssize_t] order

    cdef Split split = Split()
    cdef double[:] distances = np.empty(idx.shape[0])
    cdef double[:] left_d = np.zeros(n_classes)
    cdef double[:] right_d = np.zeros(n_classes)

    cdef double threshold, impurity
    cdef Py_ssize_t threshold_index
    cdef double[:] shapelet
    
    cdef Py_ssize_t i
    for i in range(n_shapelets):
        shapelet = draw_shapelet(idx, x, random_state)
        sliding_distance(shapelet, x, idx, out=distances)
        order = np.argsort(distances)
        partition(idx,
                  order,
                  distances,
                  y,
                  n_classes,
                  left_d,
                  right_d,
                  &threshold_index,
                  &threshold,
                  &impurity)
                
        if impurity < best_impurity:
            best_impurity = impurity
            split.threshold_index = threshold_index
            split.order = order
            split.threshold = threshold
            split.shapelet = shapelet
    # left = idx[best_order[:best_threshold_index]]
    # right = idx[best_order[best_threshold_index:]]
    
    return split

class Node:
    def prnt(self, indent):
        pass

class Branch(Node):
    def __init__(self, left, right, shapelet, threshold):
        self.left = left
        self.right = right
        self.shapelet = np.asarray(shapelet)
        self.threshold = threshold

    def prnt(self, indent=1):
        print("-" * indent, "branch:")
        print("-" * indent, " shapelet: ", self.shapelet)
        print("-" * indent, " threshold: ", self.threshold)
        print("-" * indent, " left:", end="\n")
        self.left.prnt(indent + 1)
        print("-" * indent, " right:", end="\n")
        self.right.prnt(indent + 1)

class Leaf(Node):
    def __init__(self, proba):
        self.proba = np.asarray(proba)

    def prnt(self, indent):
        print("-" * indent, "leaf: ")
        print("-" * indent, " proba: ", self.proba)

cdef class TreeBuilder:
    cdef int _n_shapelets
    cdef _random_state
    def __init__(self, int n_shapelets, object random_state):
        self._n_shapelets = n_shapelets
        self._random_state = random_state

    def build_tree(self,
                   long[:] idx,
                   double[:, :] x,
                   long[:] y,
                   Py_ssize_t n_classes):
        distribution = np.asarray(get_class_distribution(idx, y, n_classes))
        if idx.shape[0] < 2 or np.sum(distribution > 0) < 2:
            distribution = get_class_distribution(idx, y, n_classes)
            return Leaf(distribution)

        # left indices smaller than threshold
        # right indices larger than threshold
        # print("finding threshold")
        cdef Split split = find_threshold(idx,
                                          x,
                                          y,
                                          n_classes,
                                          self._n_shapelets,
                                          self._random_state)

        cdef np.ndarray[long] order = np.asarray(split.order)
        cdef long[:] left = np.empty(split.threshold_index, dtype=long)
        cdef long[:] right = np.empty(idx.shape[0] - split.threshold_index, dtype=long)

        # memcpy(left, samples, sizeof(size_t) * split.threshold_index)
        # memcpy(right, samples + split.threshold_index, n_samples - split.threshold_index)

        cdef Py_ssize_t i
        for i in range(split.threshold_index):
            left[i] = idx[order[i]]

        for i in range(idx.shape[0] - split.threshold_index):
            right[i] = idx[order[i + split.threshold_index]]
            
#        print(left.shape[0], right.shape[0])
        if left.shape[0] > 0 and right.shape[0] > 0:
            left_node = self.build_tree(left, x, y, n_classes)
            right_node = self.build_tree(right, x, y, n_classes)
            return Branch(left_node, right_node, split.shapelet, split.threshold)
        else:
            print("what?")
            distribution = get_class_distribution(idx, y, n_classes)
            return Leaf(distribution)
