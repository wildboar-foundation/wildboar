from __future__ import print_function

import numpy as np

cimport numpy as np
cimport cython

from libc.math cimport fabs, log2, INFINITY
from libc.stdlib cimport malloc, free

from numpy.random import RandomState

from pypf._distribution cimport get_class_distribution
from pypf._sliding_distance cimport sliding_distance, SlidingDistance, Shapelet
from pypf._impurity cimport safe_info

from pypf._utils cimport label_distribution, intp_ndarray_to_size_t_ptr


import numpy as np
cimport numpy as np
cimport cython
import time

def test(X, i):
    cdef SlidingDistance d = SlidingDistance(X)
    cdef Shapelet s
    s.index = 0
    s.start = 0
    s.length = 100
    d.shapelet_statistics(&s)
    cdef size_t* indicies = <size_t*> malloc(sizeof(size_t) * i.shape[0])
    intp_ndarray_to_size_t_ptr(i, indicies)
    cdef np.ndarray[np.float64_t] result = np.empty(X.shape[0])
    c = time.time()
    d.distance_list(s, indicies, i.shape[0], <double*>result.data)
    print((time.time() - c) * 1000)
    free(indicies)
    

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
