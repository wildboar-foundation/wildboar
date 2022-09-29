# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef void _mass_distance(
    double *x,
    Py_ssize_t x_length,
    double *y,
    Py_ssize_t y_length,
    double mean,
    double std,
    double *mean_x,    
    double *std_x,     
    complex *y_buffer, 
    complex *x_buffer, 
    double *dist,      
) nogil