# Authors: Isak Samsten
# License: BSD 3 clause

import operator
import re

import numpy as np

from ..utils.validation import check_option

__all__ = [
    "make_dict_filter",
    "make_str_filter",
    "make_list_filter",
    "make_filter",
]


def __n_dims_filter(verb, dataset, x, y):
    n_dims = x.shape[1] if x.ndim > 2 else 1
    return __verb_compare(verb, n_dims)


def __casting_op(op, lhs, rhs):
    lhs_class = lhs.__class__
    return op(lhs, lhs_class(rhs))


def __verb_compare(verb, value):
    match = re.match(__VERB_PATTERN, verb)
    if match:
        return _OPERATORS[match.group(1)](value, match.group(2))
    else:
        raise ValueError(
            "Invalid comparision %s, must match %s" % (verb, __VERB_PATTERN)
        )


def __new_composite_filter(filters):
    def f(dataset, x, y):
        for filter in filters:
            if not filter(dataset, x, y):
                return False
        return True

    return f


def make_filter(filter):
    """Create a new filter

    Parameters
    ----------
    filter : str, list or dict
        The filter

    Returns
    -------
    function
        The filter function
    """
    if isinstance(filter, list):
        return make_list_filter(filter)
    elif isinstance(filter, dict):
        return make_dict_filter(filter)
    elif isinstance(filter, str):
        return make_str_filter(filter)
    else:
        raise TypeError(
            "filter must be list, dict or str, not %s" % type(filter).__qualname__
        )


def make_list_filter(filter):
    """Make a new filter based on a list of filter strings

    Parameters
    ----------
    filter : list
        A list of filter strings

    Returns
    -------
    function
        The filter function
    """
    filters = []
    for str_filter in filter:
        filters.append(make_str_filter(str_filter))

    return __new_composite_filter(filters)


def make_dict_filter(filter):
    """Make a new filter

    Parameters
    ----------
    filter : dict
        The dict of [subject] -> [op][verb]

    Returns
    -------
    function
        The filter function
    """
    filters = []
    for subject, verb in filter.items():
        filters.append(
            lambda dataset, x, y: check_option(_SUBJECTS, subject, "subject")(
                verb, dataset, x, y
            )
        )

    return __new_composite_filter(filters)


def make_str_filter(filter):
    """Make a new filter

    Parameters
    ----------
    filter : str
        The filter string [subject][op][verb]

    Returns
    -------
    function
        The filter function
    """
    match = re.match(__SUBJECT_VERB_PATTERN, filter)
    if match:
        subject = match.group(1)
        verb = match.group(2)
    else:
        raise ValueError(
            "Invalid filter %s, must match %s" % (filter, __SUBJECT_VERB_PATTERN)
        )

    def f(dataset, x, y):
        return check_option(_SUBJECTS, subject, "subject")(verb, dataset, x, y)

    return f


__VERB_PATTERN = re.compile(r"^(<|<=|>=|>|=|=~)\s*(\w+)$")
__SUBJECT_VERB_PATTERN = re.compile(r"^(\w+)\s*((?:<|<=|>=|>|=|=~)\s*\w+)$")


_SUBJECTS = {
    "dataset": lambda verb, dataset, x, y: __verb_compare(verb, dataset),
    "n_samples": lambda verb, dataset, x, y: __verb_compare(verb, x.shape[0]),
    "n_timestep": lambda verb, dataset, x, y: __verb_compare(verb, x.shape[-1]),
    "n_dims": __n_dims_filter,
    "n_labels": lambda verb, dataset, x, y: __verb_compare(verb, np.unique(y).shape[0]),
}

_OPERATORS = {
    "=~": lambda x, y: y in x,
    "=": lambda x, y: __casting_op(operator.eq, x, y),
    "<=": lambda x, y: __casting_op(operator.le, x, y),
    ">=": lambda x, y: __casting_op(operator.ge, x, y),
    "<": lambda x, y: __casting_op(operator.lt, x, y),
    ">": lambda x, y: __casting_op(operator.gt, x, y),
}
