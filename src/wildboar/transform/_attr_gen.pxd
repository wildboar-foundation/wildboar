# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

from numpy cimport uint32_t

from wildboar.utils cimport TSArray


cdef struct Attribute:
    Py_ssize_t dim       # the dimension of the attribute -1 for undefined
    void* attribute        # the attribute value


cdef class AttributeGenerator:

    # Reset the (allocated) attributes of the
    cdef int reset(self, TSArray X) noexcept nogil

    # Safe to call without calling reset first
    cdef Py_ssize_t get_n_attributess(self, TSArray X) noexcept nogil

    # Safe to call without calling reset first
    cdef Py_ssize_t get_n_outputs(self, TSArray X) noexcept nogil

    # Requires `reset`
    cdef Py_ssize_t next_attribute(
        self,
        Py_ssize_t attribute_id,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Attribute *transient,
        uint32_t *seed
    ) noexcept nogil

    # Initialize a persisent attribute from a transient attribute
    #
    # NOTE: We permit moving of ownership of `transient.attribute`.
    #       If is unsafe to use `transient.attribute` after `init_persistent`
    #       has been called.
    # Requires: `reset`
    cdef Py_ssize_t init_persistent(
        self,
        TSArray X,
        Attribute *transient,
        Attribute *persistent
    ) noexcept nogil

    cdef Py_ssize_t free_transient(self, Attribute *attribute) noexcept nogil

    cdef Py_ssize_t free_persistent(self, Attribute *attribute) noexcept nogil

    # Calculate the attribute value for sample using the transient attribute
    cdef double transient_value(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil

    cdef Py_ssize_t transient_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_attribute,
    ) noexcept nogil

    # Calculate the attribute value for sample using the persistent attribute
    cdef double persistent_value(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil

    cdef Py_ssize_t persistent_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_attribute,
    ) noexcept nogil

    # Calculate the attribute value for all samples using the transient featuer
    cdef void transient_values(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double* values
    ) noexcept nogil

    # Calculate the attribute value for all samples using the persistent attribute
    cdef void persistent_values(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double* values
    ) noexcept nogil

    cdef object persistent_to_object(self, Attribute *attribute)

    cdef Py_ssize_t persistent_from_object(self, object object, Attribute *attribute)
