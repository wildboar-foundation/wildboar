# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

from numpy cimport uint32_t

from wildboar.utils cimport TSArray


cdef struct Attribute:
    Py_ssize_t dim       # the dimension of the attribute -1 for undefined
    void* attribute        # the attribute value


cdef class AttributeGenerator:
    cdef Py_ssize_t n_dims
    cdef Py_ssize_t n_timestep

    # Reset the (allocated) attributes of the generator
    # by default delegates to _reset and sets n_dims and n_timestep
    cdef int reset(self, TSArray X) noexcept nogil

    cdef int _reset(self, TSArray X) noexcept nogil

    # Get the number of attributes *during* fitting, since the number of
    # attributes might depend on the number of samples.
    cdef Py_ssize_t get_n_attributes(
        self, Py_ssize_t* samples, Py_ssize_t n_samples
    ) noexcept nogil

    # Get the number of outputs *during* fitting, since the number of outputs
    # might depend on the number of samples. The number of outputs and
    # attributes might differ (each attribute can be used to
    # compute multiple outputs).
    cdef Py_ssize_t get_n_outputs(
        self, Py_ssize_t *samples, Py_ssize_t n_samples
    ) noexcept nogil

    # Requires `reset`. Get the next attribute.
    # attribute_id must be between 0 and get_n_attributes(samples, n_samples).
    cdef Py_ssize_t next_attribute(
        self,
        Py_ssize_t attribute_id,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Attribute *transient,
        uint32_t *seed
    ) noexcept nogil

    # Initialize a persistent attribute from a transient attribute
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

    # Calculate the attribute value for all samples using the transient feature
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

    # Move the attribute to Python. The return is a tuple (dim, obj), where the
    # value of obj depends on the implementation.
    cdef object persistent_to_object(self, Attribute *attribute)

    # Move the attribute from Python. The `object` argument is a tuple (dim,
    # obj) where obj is returned bu `persistent_to_object`.
    cdef Py_ssize_t persistent_from_object(self, object object, Attribute *attribute)
