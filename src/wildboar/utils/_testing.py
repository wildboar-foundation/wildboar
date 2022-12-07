# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from sklearn import clone
from sklearn.utils._param_validation import (
    generate_invalid_param_val,
    generate_valid_param,
    make_constraint,
)

from ..base import BaseEstimator

_DUMMY_X = np.zeros((10, 10))
_DUMMY_X.setflags(write=False)
_DUMMY_Y = np.zeros(10)
_DUMMY_Y[:5] = 1
_DUMMY_Y.setflags(write=False)


def assert_exhaustive_parameter_checks(estimator: BaseEstimator):
    assert hasattr(estimator.__class__, "_parameter_constraints")
    assert (
        estimator.get_params(deep=False).keys()
        == estimator.__class__._parameter_constraints.keys()
    )


def assert_parameter_checks(estimator: BaseEstimator):
    assert hasattr(estimator.__class__, "_parameter_constraints")
    for param, constraints in estimator.__class__._parameter_constraints.items():
        for constraint in (make_constraint(constraint) for constraint in constraints):
            try:
                invalid_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                continue

            valid_value = generate_valid_param(constraint)

            estimator_ = clone(estimator)
            estimator_.set_params(**{param: invalid_value})
            with pytest.raises(ValueError):
                estimator_.fit(_DUMMY_X, _DUMMY_Y)

            estimator_ = clone(estimator)
            estimator_.set_params(**{param: valid_value})
            estimator_.fit(_DUMMY_X, _DUMMY_Y)
