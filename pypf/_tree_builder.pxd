from pypf._sliding_distance cimport ShapeletInfo

cdef struct SplitPoint:
   size_t split_point
   double threshold
   ShapeletInfo shapelet_info
