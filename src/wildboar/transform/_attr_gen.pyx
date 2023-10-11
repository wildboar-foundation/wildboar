# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

from libc.math cimport NAN
from libc.stdlib cimport free, malloc
from numpy cimport uint32_t

from ..utils cimport TSArray


cdef class AttributeGenerator:

    cdef int reset(self, TSArray X) noexcept nogil:
        return 0

    cdef Py_ssize_t get_n_attributess(self, TSArray X) noexcept nogil:
        return -1

    cdef Py_ssize_t get_n_outputs(self, TSArray X) noexcept nogil:
        return self.get_n_attributess(X)

    cdef Py_ssize_t next_attribute(
        self,
        Py_ssize_t attribute_id,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Attribute *transient,
        uint32_t *seed,
    ) noexcept nogil:
        return -1

    cdef Py_ssize_t free_transient(self, Attribute *attribute) noexcept nogil:
        return -1

    cdef Py_ssize_t free_persistent(self, Attribute *attribute) noexcept nogil:
        return -1

    cdef Py_ssize_t init_persistent(
        self,
        TSArray X,
        Attribute *transient,
        Attribute *persistent
    ) noexcept nogil:
        return 0

    cdef double transient_value(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        return NAN

    cdef double persistent_value(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        return NAN

    cdef Py_ssize_t transient_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_attribute,
    ) noexcept nogil:
        return -1

    cdef Py_ssize_t persistent_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_attribute,
    ) noexcept nogil:
        return -1

    cdef object persistent_to_object(self, Attribute *attribute):
        return None

    cdef Py_ssize_t persistent_from_object(self, object object, Attribute *attribute):
        return 0

    cdef void transient_values(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double* values
    ) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n_samples):
            values[i] = self.transient_value(attribute, X, samples[i])

    cdef void persistent_values(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double* values
    ) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n_samples):
            values[i] = self.persistent_value(attribute, X, samples[i])
