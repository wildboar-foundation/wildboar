import warnings
from functools import partial
from unittest.case import SkipTest

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import SkipTestWarning
from sklearn.utils._testing import (
    assert_allclose_dense_sparse,
    ignore_warnings,
    set_random_state,
)
from sklearn.utils.estimator_checks import _enforce_estimator_tags_y, _maybe_skip
from sklearn.utils.estimator_checks import (
    _yield_all_checks as _yield_all_checks_sklearn,
)
from sklearn.utils.validation import has_fit_parameter


def _yield_all_checks(estimator):
    if has_fit_parameter(estimator, "sample_weight"):
        yield partial(check_sample_weights_invariance_samples_order, kind="ones")
        yield partial(check_sample_weights_invariance_samples_order, kind="zeros")


def check_estimator(estimator, generate_only=False, ignore=None):
    """Check if estimator adheres to scikit-learn (and wildboar) conventions.

    This method delegates to `check_estimator` in scikit-learn but monkey-patches
    the estimator with tags to skip some tests related to performance.

    We also add a new parameter to allow for silently ignoring some scikit-learn
    tests.

    Finally, we also add a suite of specific wildboar-tests tailored towards
    time series classifiers, regressors and transformers.

    Parameters
    ----------
    estimator : estimator object
        Estimator instance to check.

    generate_only : bool, default=False
        When `False`, checks are evaluated when `check_estimator` is called.
        When `True`, `check_estimator` returns a generator that yields
        (estimator, check) tuples. The check is run by calling
        `check(estimator)`.

    ignore : list, optional
        Ignore the checks in the list.

    Returns
    -------
    checks_generator : generator
        Generator that yields (estimator, check) tuples. Returned when
        `generate_only=True`.

    """

    if ignore is None:
        ignore = []

    name = type(estimator).__name__

    def checks_generator():
        if estimator is not None:
            if hasattr(estimator, "_more_tags"):
                _more_tags = estimator._more_tags().copy()
                _more_tags.update({"poor_score": True})

            old_more_tags = estimator.__class__._more_tags

            def _new_more_tags_skip_low_score(self):
                return _more_tags

            # Monkey-patch the estimator to always have the tag "poor_score" set
            # to true. We have specific tests in wildboar for testing performance.
            setattr(estimator.__class__, "_more_tags", _new_more_tags_skip_low_score)

        for check in _yield_all_checks_sklearn(estimator):
            check_name = (
                check.func.__name__ if isinstance(check, partial) else check.__name__
            )

            # Silently ignore any scikit-learn tess in the ignore list
            if check_name not in ignore:
                check = _maybe_skip(estimator, check)
                yield estimator, partial(check, name)

        for check in _yield_all_checks(estimator):
            check = _maybe_skip(estimator, check)
            yield estimator, partial(check, name)

        if estimator is not None:
            # Reset the old _more_tags() method
            setattr(estimator.__class__, "_more_tags", old_more_tags)

    if generate_only:
        return checks_generator()

    for estimator, check in checks_generator():
        try:
            check(estimator)
        except SkipTest as exception:
            warnings.warn(str(exception), SkipTestWarning)


@ignore_warnings(category=FutureWarning)
def check_sample_weights_invariance_samples_order(name, estimator_orig, kind="ones"):
    # For kind="ones" check that the estimators yield same results for
    # unit weights and no weights
    # For kind="zeros" check that setting sample_weight to 0 is equivalent
    # to removing corresponding samples.
    estimator1 = clone(estimator_orig)
    estimator2 = clone(estimator_orig)
    set_random_state(estimator1, random_state=0)
    set_random_state(estimator2, random_state=0)

    X1 = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.float64,
    )
    y1 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)

    if kind == "ones":
        X2 = X1
        y2 = y1
        sw2 = np.ones(shape=len(y1))
        err_msg = (
            f"For {name} sample_weight=None is not equivalent to sample_weight=ones"
        )
    elif kind == "zeros":
        # Construct a dataset that is very different to (X, y) if weights
        # are disregarded, but identical to (X, y) given weights.
        X2 = np.vstack([X1, X1 + 1])
        y2 = np.hstack([y1, 3 - y1])
        sw2 = np.ones(shape=len(y1) * 2)
        sw2[len(y1) :] = 0

        # NOTE: We disable shuffling to allow support estimators that sample
        #
        # X2, y2, sw2 = shuffle(X2, y2, sw2, random_state=0)

        err_msg = (
            f"For {name}, a zero sample_weight is not equivalent to removing the sample"
        )
    else:  # pragma: no cover
        raise ValueError

    y1 = _enforce_estimator_tags_y(estimator1, y1)
    y2 = _enforce_estimator_tags_y(estimator2, y2)

    estimator1.fit(X1, y=y1, sample_weight=None)
    estimator2.fit(X2, y=y2, sample_weight=sw2)

    for method in ["predict", "predict_proba", "decision_function", "transform"]:
        if hasattr(estimator_orig, method):
            X_pred1 = getattr(estimator1, method)(X1)
            X_pred2 = getattr(estimator2, method)(X1)
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg)
