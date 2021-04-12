# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten
import re
import operator

import numpy as np

__all__ = [
    "make_dict_filter",
    "make_str_filter",
    "make_list_filter",
    "make_filter",
]


def __n_dims_filter(verb, dataset, x, y):
    if x.ndims < 3:
        return True
    else:
        return __verb_compare(verb, x.shape[1])


def __casting_op(op, l, r):
    lhs_class = l.__class__
    return op(l, lhs_class(r))


def __verb_compare(verb, value):
    match = re.match(__VERB_PATTERN, verb)
    if match:
        return _OPERATORS[match.group(1)](value, match.group(2))
    else:
        raise ValueError("invalid verb (%s)" % verb)


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
        return make_dict_filter(dict)
    elif isinstance(filter, str):
        return make_str_filter(filter)
    else:
        raise ValueError("invalid filter (%r)" % filter)


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
        if subject not in _SUBJECTS:
            raise ValueError("invalid subject (%s)" % subject)
        filters.append(lambda dataset, x, y: _SUBJECTS[subject](verb, dataset, x, y))

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
        if subject not in _SUBJECTS:
            raise ValueError("invalid subject (%s)" % subject)
    else:
        raise ValueError("invalid filter (%s)" % filter)

    def f(dataset, x, y):
        return _SUBJECTS[subject](verb, dataset, x, y)

    return f


__VERB_PATTERN = re.compile("^(<|<=|>=|>|=|=~)\s*(\w+)$")
__SUBJECT_VERB_PATTERN = re.compile("^(\w+)\s*((?:<|<=|>=|>|=|=~)\s*\w+)$")


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
