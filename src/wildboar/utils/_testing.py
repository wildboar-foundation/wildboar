# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from sklearn import clone
from sklearn.utils._param_validation import (
    InvalidParameterError,
    generate_invalid_param_val,
    generate_valid_param,
    make_constraint,
)
from sklearn.utils._testing import MinimalClassifier

from ..base import BaseEstimator, is_explainer

_DUMMY_X = np.zeros((10, 10))
_DUMMY_X.setflags(write=False)
_DUMMY_Y = np.zeros(10)
_DUMMY_Y[:5] = 1
_DUMMY_Y.setflags(write=False)


def assert_exhaustive_parameter_checks(estimator: BaseEstimator):
    """
    Assert that all parameter are checked.

    Parameters
    ----------
    estimator : BaseEstimator
       The estimator to check.

    Attributes
    ----------
    test : bool
        Ok.
    """
    assert hasattr(estimator.__class__, "_parameter_constraints")
    assert (
        estimator.get_params(deep=False).keys()
        == estimator.__class__._parameter_constraints.keys()
    )


def assert_parameter_checks(estimator: BaseEstimator, skip=None):
    """
    Assert that all parameter checks are correct.

    Extended.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator.
    skip : list, optional
        The parameter constraints to skip.
    """
    assert hasattr(estimator.__class__, "_parameter_constraints")
    if is_explainer(estimator):
        clf = MinimalClassifier()
        clf.fit(_DUMMY_X, _DUMMY_Y)
        clf.n_features_in_ = _DUMMY_X.shape[1]

    for param, constraints in estimator.__class__._parameter_constraints.items():
        if skip is not None and param in skip:
            continue

        for constraint in (make_constraint(constraint) for constraint in constraints):
            try:
                invalid_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                invalid_value = None

            if invalid_value is not None:
                estimator_ = clone(estimator)
                estimator_.set_params(**{param: invalid_value})
                with pytest.raises(InvalidParameterError):
                    if is_explainer(estimator):
                        try:
                            estimator_._validate_params()
                        except Exception as e:
                            if isinstance(e, InvalidParameterError):
                                raise e
                    else:
                        try:
                            estimator_._validate_params()
                        except Exception as e:
                            if isinstance(e, InvalidParameterError):
                                raise e

            try:
                valid_value = generate_valid_param(constraint)
            except Exception:
                continue

            estimator_ = clone(estimator)
            estimator_.set_params(**{param: valid_value})
            if is_explainer(estimator):
                try:
                    estimator_._validate_params()
                except Exception as e:
                    if isinstance(e, InvalidParameterError):
                        raise e

            else:
                try:
                    estimator_._validate_params()
                except Exception as e:
                    if isinstance(e, InvalidParameterError):
                        raise e
