# Authors: Isak Samsten
# License: BSD 3 clause

import warnings
from functools import partial
from unittest.case import SkipTest

import numpy as np
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import SkipTestWarning
from sklearn.utils._tags import get_tags
from sklearn.utils._testing import (
    assert_allclose_dense_sparse,
    ignore_warnings,
    raises,
    set_random_state,
)
from sklearn.utils.estimator_checks import (
    _enforce_estimator_tags_y,
    _maybe_mark,
)
from sklearn.utils.estimator_checks import (
    estimator_checks_generator as estimator_checks_generator_sklearn,
)
from sklearn.utils.validation import has_fit_parameter

from ..base import is_counterfactual, is_explainer


def _yield_classifier_test(estimator):
    if has_fit_parameter(estimator, "sample_weight"):
        yield partial(_check_sample_weights_invariance_samples_order, kind="ones")
        yield partial(_check_sample_weights_invariance_samples_order, kind="zeros")

    # if not _safe_tags(estimator).get("poor_score", False):
    #     yield check_decent_score


def _yield_regressor_checks(estimator):
    pass


def _yield_transform_checks(estimator):
    pass


def _yield_explainer_checks(estimator):
    pass


def _yield_all_checks(estimator):
    if is_classifier(estimator):
        for check in _yield_classifier_test(estimator):
            yield check

    if is_regressor(estimator):
        pass
    #     for check in _yield_regressor_checks(estimator):
    #         yield check

    if hasattr(estimator, "transform"):
        pass
    #     for check in _yield_transform_checks(estimator):
    #         yield check

    if is_explainer(estimator):
        pass
    #     for check in _yield_explainer_checks(estimator):
    #         yield check

    if is_counterfactual(estimator):
        pass

    if not is_explainer(estimator):
        if hasattr(estimator, "estimator_params"):
            yield _check_consistent_estimator_params

        if get_tags(estimator).input_tags.three_d_array:
            yield _check_force_n_dims_raises
            yield _check_force_n_dims


# noqa: H0002
def check_estimator(
    estimator, expected_failed_checks=None, mark="skip", ignore=None, skip_scikit=False
):
    """
    Check if estimator adheres to scikit-learn (and wildboar) conventions.

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
    expected_failed_checks : dict, optional
        A dictionary of the form::

            {
                "check_name": "this check is expected to fail because ...",
            }

        Where `"check_name"` is the name of the check, and `"my reason"` is why
        the check fails.

        .. versionadded:: 1.3
    mark : {"skip"} or None, optional
        Marking a test as "skip" is done via wrapping the check in a function
        that raises a :class:`~sklearn.exceptions.SkipTest` exception.
    ignore : list, optional
        Ignore the checks in the list.

        .. deprecated:: 1.3
    skip_scikit : bool, optional
        Skip all scikit-learn tests.

    Returns
    -------
    generator
        Generator that yields (estimator, check) tuples. Returned when
        `generate_only=True`.
    """

    if ignore is None:
        ignore = []

    name = type(estimator).__name__

    def checks_generator():
        if not skip_scikit:
            if estimator is not None:
                if hasattr(estimator, "__sklearn_tags__"):
                    __sklearn_tags__ = estimator.__sklearn_tags__()
                    if __sklearn_tags__.classifier_tags is not None:
                        __sklearn_tags__.classifier_tags.poor_score = True
                    if __sklearn_tags__.regressor_tags is not None:
                        __sklearn_tags__.regressor_tags.poor_score = True

                old__sklearn_tags__ = estimator.__class__.__sklearn_tags__

                def _new_sklearn_tags_skip_low_score(self):
                    return __sklearn_tags__

                # Monkey-patch the estimator to always have the tag "poor_score" set
                # to true. We have specific tests in wildboar for testing performance.
                setattr(
                    estimator.__class__,
                    "__sklearn_tags__",
                    _new_sklearn_tags_skip_low_score,
                )

            for _, check in estimator_checks_generator_sklearn(
                estimator,
                legacy=True,
                expected_failed_checks=expected_failed_checks,
                mark=mark,
            ):
                check_name = (
                    check.func.__name__
                    if isinstance(check, partial)
                    else check.__name__
                )

                # Silently ignore any scikit-learn tess in the ignore list
                if check_name not in ignore:
                    yield estimator, check

            if estimator is not None:
                setattr(estimator.__class__, "__sklearn_tags__", old__sklearn_tags__)

        for check in _yield_all_checks(estimator):
            yield _maybe_mark(
                estimator,
                partial(check, name),
                expected_failed_checks=expected_failed_checks,
                mark=mark,
            )

    for estimator, check in checks_generator():
        try:
            check(estimator)
        except SkipTest as exception:
            warnings.warn(str(exception), SkipTestWarning)


@ignore_warnings(category=FutureWarning)
def _check_sample_weights_invariance_samples_order(name, estimator_orig, kind="ones"):
    # For kind="ones" check that the estimators yield same results for
    # unit weights and no weights
    # For kind="zeros" check that setting sample_weight to 0 is equivalent
    # to removing corresponding samples.
    estimator1 = clone(estimator_orig)
    estimator2 = clone(estimator_orig)
    if hasattr(estimator_orig, "bootstrap"):
        estimator1.bootstrap = False
        estimator2.bootstrap = False

    set_random_state(estimator1, random_state=0)
    set_random_state(estimator2, random_state=0)

    X1, y1 = _dummy_dataset()

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


def _dummy_dataset():
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
    return X1, y1


def _dataset_shape(*shape):
    return np.arange(np.prod(shape)).reshape(*shape), np.zeros(shape[0])


def _check_consistent_estimator_params(name, estimator):
    x, y = _dummy_dataset()
    estimator = clone(estimator)
    estimator.fit(x, y)

    for tree in estimator.estimators_:
        for param in estimator.estimator_params:
            assert getattr(tree, param) == getattr(estimator, param), (
                f"For {name} estimator_params are not equivalent "
                "to the base estimators parameters"
            )


def _check_force_n_dims_raises(name, estimator):
    X1, y1 = _dataset_shape(10, 3, 10)
    X2, y2 = _dataset_shape(10, 10)

    estimator = clone(estimator)
    estimator._force_n_dims = 2
    match = "has _force_n_dims set to 2"
    err_msg = (
        f"If _force_n_dims is set the {name} must reject "
        "3d arrays with unsupported n_dims."
    )

    with raises(
        ValueError,
        match=match,
        err_msg=err_msg,
    ):
        estimator.fit(X1, y1)

    if has_fit_parameter(estimator, "check_input"):
        with raises(ValueError, match=match, err_msg=err_msg):
            estimator.fit(X1, y1, check_input=False)

    del estimator._force_n_dims
    estimator.fit(X2, y2)
    estimator._force_n_dims = 2
    for method in ["predict", "predict_proba", "decision_function", "transform"]:
        if hasattr(estimator, method):
            with raises(ValueError, match=match, err_msg=err_msg):
                getattr(estimator, method)(X1)


def _check_force_n_dims(name, estimator):
    X1, y1 = _dataset_shape(10, 3, 10)
    X2, y2 = _dataset_shape(10, 3 * 10)
    estimator = clone(estimator)
    estimator.fit(X1, y1)

    estimator._force_n_dims = 3
    for method in ["predict", "predict_proba", "decision_function", "transform"]:
        if hasattr(estimator, method):
            getattr(estimator, method)(X2)
