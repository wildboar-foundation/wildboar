# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal

from wildboar.datasets import load_gun_point
from wildboar.tree import (
    ExtraShapeletTreeClassifier,
    ExtraShapeletTreeRegressor,
    ShapeletTreeClassifier,
    ShapeletTreeRegressor,
)
from wildboar.utils._testing import (
    assert_exhaustive_parameter_checks,
    assert_parameter_checks,
)
from wildboar.utils.estimator_checks import check_estimator


@pytest.mark.parametrize(
    "clf",
    [
        ShapeletTreeClassifier(strategy="random", n_shapelets=10),
        ShapeletTreeRegressor(strategy="random", n_shapelets=10),
        ExtraShapeletTreeClassifier(n_shapelets=10),
        ExtraShapeletTreeRegressor(n_shapelets=10),
    ],
)
def test_check_estimator(clf):
    assert_exhaustive_parameter_checks(clf)
    assert_parameter_checks(clf, skip=["alpha", "impurity_equality_tolerance"])
    check_estimator(
        clf,
        expected_failed_checks={
            "check_sample_weight_equivalence_on_dense_data": "not working",
        },
        mark="skip",
    )


def test_shapelet_tree_strategy_best():
    X, y = load_gun_point()
    t = ShapeletTreeClassifier(
        strategy="best",
        shapelet_size=0.1,
        sample_size=0.1,
        n_shapelets=1,
        metric="scaled_euclidean",
        random_state=1,
    ).fit(X, y)
    expected_left = np.array([1, 2, -1, 4, -1, -1, 7, -1, 9, -1, 11, -1, -1])
    expected_right = np.array([6, 3, -1, 5, -1, -1, 8, -1, 10, -1, 12, -1, -1])

    # fmt: off
    expected_threshold = np.array(
            [0.78738805, 2.23066515, 0.        , 1.83841356, 0.        ,
             0.        , 2.67736822, 0.        , 2.71344458, 0.        ,
             0.45726085, 0.        , 0.        ]
    )
    # fmt: on
    assert_almost_equal(t.tree_.left, expected_left)
    assert_almost_equal(t.tree_.right, expected_right)
    assert_almost_equal(
        t.tree_.threshold[t.tree_.left != -1], expected_threshold[expected_left != -1]
    )


def test_shapelet_tree_apply():
    x_train, x_test, y_train, y_test = load_gun_point(False)
    f = ShapeletTreeClassifier(n_shapelets=10, strategy="random", random_state=123)
    f.fit(x_test, y_test)
    actual_apply = f.apply(x_train)
    # fmt:off
    expected_apply = np.array(
        [8, 12,  8,  8, 12, 12, 20, 20, 20,  2,  2,  2,  2,  2, 12,  2, 16,
        12,  2, 20,  2,  2,  2, 20,  8, 20,  2,  2, 20, 20,  2, 20, 20,  8,
        12,  8, 20,  8, 20, 20, 14,  2,  2,  2, 20, 12,  8, 20,  8, 20],
        dtype=int,
    )
    # fmt: on
    assert actual_apply.dtype == np.intp
    assert_array_equal(actual_apply, expected_apply)


def test_shapelet_tree_decision_path():
    x_train, x_test, y_train, y_test = load_gun_point(False)
    f = ShapeletTreeClassifier(n_shapelets=10, strategy="random", random_state=123)
    f.fit(x_test, y_test)
    actual_decision_path = f.decision_path(x_train)
    expected_decision_path = np.zeros((50, 21), dtype=bool)
    # fmt:off
    true_indicies = (
            np.array([ 0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,
                       4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  6,  6,  7,  7,  8,  8,  9,
                       9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 14, 14, 14, 15, 15, 16,
                       16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 19, 19, 20, 20,
                       21, 21, 22, 22, 23, 23, 24, 24, 24, 24, 25, 25, 26, 26, 27, 27, 28,
                       28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34,
                       34, 35, 35, 35, 35, 36, 36, 37, 37, 37, 37, 38, 38, 39, 39, 40, 40,
                       40, 40, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 45, 45, 45,
                       46, 46, 46, 46, 47, 47, 48, 48, 48, 48, 49, 49]),
            np.array([ 0,  1,  3,  4,  0,  1,  3,  9, 11,  0,  1,  3,  4,  0,  1,  3,  4,
                       0,  1,  3,  9, 11,  0,  1,  3,  9, 11,  0, 18,  0, 18,  0, 18,  0,
                       1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  3,  9, 11,  0,  1,  0,
                       1,  3,  9, 11, 13, 15,  0,  1,  3,  9, 11,  0,  1,  0, 18,  0,  1,
                       0,  1,  0,  1,  0, 18,  0,  1,  3,  4,  0, 18,  0,  1,  0,  1,  0,
                       18,  0, 18,  0,  1,  0, 18,  0, 18,  0,  1,  3,  4,  0,  1,  3,  9,
                       11,  0,  1,  3,  4,  0, 18,  0,  1,  3,  4,  0, 18,  0, 18,  0,  1,
                       3,  9, 11, 13,  0,  1,  0,  1,  0,  1,  0, 18,  0,  1,  3,  9, 11,
                       0,  1,  3,  4,  0, 18,  0,  1,  3,  4,  0, 18])
    )
    # fmt:on
    expected_decision_path[true_indicies] = True
    assert actual_decision_path.dtype == np.bool_
    assert_array_equal(actual_decision_path.toarray(), expected_decision_path)


@pytest.mark.parametrize(
    "criterion,expected_left,expected_right,threshold",
    [
        pytest.param(
            "entropy",
            [
                1,
                2,
                -1,
                4,
                5,
                -1,
                7,
                8,
                -1,
                10,
                11,
                12,
                13,
                -1,
                -1,
                16,
                -1,
                18,
                -1,
                20,
                -1,
                -1,
                -1,
                -1,
                25,
                -1,
                27,
                28,
                29,
                -1,
                31,
                -1,
                -1,
                -1,
                -1,
                -1,
                37,
                38,
                -1,
                40,
                41,
                -1,
                -1,
                44,
                -1,
                -1,
                -1,
            ],
            [
                36,
                3,
                -1,
                35,
                6,
                -1,
                24,
                9,
                -1,
                23,
                22,
                15,
                14,
                -1,
                -1,
                17,
                -1,
                19,
                -1,
                21,
                -1,
                -1,
                -1,
                -1,
                26,
                -1,
                34,
                33,
                30,
                -1,
                32,
                -1,
                -1,
                -1,
                -1,
                -1,
                46,
                39,
                -1,
                43,
                42,
                -1,
                -1,
                45,
                -1,
                -1,
                -1,
            ],
            [
                2.3858733725835246,
                1.3030318030184422,
                1.270966228821233,
                0.571506183038313,
                7.163849443826518,
                0.4043070891203555,
                3.364997328797938,
                7.144333912205175,
                0.8748218549098081,
                0.23495720569659587,
                0.8174078818698833,
                1.8713786644191401,
                2.2760411164401315,
                5.2621440059644575,
                3.914107922505309,
                1.9634446959659837,
                1.809236776082022,
                6.981438166500511,
                3.1568984289668087,
                0.7390856063408281,
                5.081020230264091,
                2.545603715179477,
                0.5935289709243529,
            ],
        ),
        pytest.param(
            "gini",
            [
                1,
                2,
                -1,
                4,
                5,
                -1,
                7,
                8,
                -1,
                10,
                11,
                12,
                13,
                -1,
                -1,
                16,
                -1,
                18,
                -1,
                20,
                -1,
                -1,
                -1,
                -1,
                25,
                -1,
                27,
                28,
                29,
                -1,
                31,
                -1,
                -1,
                -1,
                -1,
                -1,
                37,
                38,
                -1,
                40,
                41,
                -1,
                -1,
                44,
                -1,
                -1,
                -1,
            ],
            [
                36,
                3,
                -1,
                35,
                6,
                -1,
                24,
                9,
                -1,
                23,
                22,
                15,
                14,
                -1,
                -1,
                17,
                -1,
                19,
                -1,
                21,
                -1,
                -1,
                -1,
                -1,
                26,
                -1,
                34,
                33,
                30,
                -1,
                32,
                -1,
                -1,
                -1,
                -1,
                -1,
                46,
                39,
                -1,
                43,
                42,
                -1,
                -1,
                45,
                -1,
                -1,
                -1,
            ],
            [
                2.3858733725835246,
                1.3030318030184422,
                1.270966228821233,
                0.571506183038313,
                7.163849443826518,
                0.4043070891203555,
                3.364997328797938,
                7.144333912205175,
                0.8748218549098081,
                0.23495720569659587,
                0.8174078818698833,
                1.8713786644191401,
                2.2760411164401315,
                5.2621440059644575,
                3.914107922505309,
                1.9634446959659837,
                1.809236776082022,
                6.981438166500511,
                3.1568984289668087,
                0.7390856063408281,
                5.081020230264091,
                2.545603715179477,
                0.5935289709243529,
            ],
        ),
    ],
)
def test_extra_tree_classifier(criterion, expected_left, expected_right, threshold):
    x, y = load_gun_point()
    f = ExtraShapeletTreeClassifier(
        n_shapelets=1, criterion=criterion, random_state=123
    )
    f.fit(x, y)
    assert (f.predict(x) == y).sum() == 191
    assert_equal(f.tree_.left, expected_left)
    assert_equal(f.tree_.right, expected_right)
    assert_equal(f.tree_.left > 0, f.tree_.right > 0)
    assert_almost_equal(f.tree_.threshold[f.tree_.left > 0], threshold)


def test_extra_tree_regressor():
    x, y = load_gun_point()
    f = ExtraShapeletTreeRegressor(
        n_shapelets=1, criterion="squared_error", random_state=123
    )
    f.fit(x, y.astype(float))
    assert_almost_equal(f.tree_.threshold[0], 2.3858733725835246)
    assert_almost_equal(f.tree_.threshold[6], 7.163849443826518)
    assert (f.predict(x) == y.astype(float)).sum() == 182


def test_impurity_equality_tolerance():
    X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)
    f = ShapeletTreeClassifier(
        impurity_equality_tolerance=0.1, strategy="random", random_state=1
    )
    f.fit(X_train, y_train)
    actual_apply = f.apply(X_test)
    # fmt:off
    expected_apply = np.array([
        3,  6,  6,  3,  6,  6,  6,  6,  2,  9,  9,  3,  6,  3,  3,  3,  6,
       10,  6,  6,  3,  6,  6,  9,  3,  6,  3,  3,  2,  3,  6,  6,  6,  6,
        6,  3,  3,  3,  3,  3,  6,  6,  3,  3,  6,  6,  6,  3, 10,  3,  6,
        6,  3,  6,  6,  3,  6,  6,  3,  6,  3,  6,  3,  3,  3,  3,  3,  3,
        6,  6,  6,  6,  2,  6,  6,  6,  6,  6,  6,  6,  6,  3,  6,  9, 10,
        6,  3,  3,  6,  6,  6,  6,  3,  6, 10,  9,  6,  6,  3,  3,  6,  3,
        3,  2,  6,  3,  6,  6,  9,  6,  6,  9,  3,  6,  3,  3,  6,  3,  6,
        6,  3,  6,  6,  6,  6,  6,  3,  3,  6,  6,  6,  6,  3,  6,  6,  6,
        3,  3,  3,  3,  6,  6,  9,  6,  9,  6,  3,  3,  6,  6]
    )
    # fmt:on
    assert_array_equal(actual_apply, expected_apply)


def test_coverage_probability(gun_point):
    X_train, X_test, y_train, y_test = gun_point
    f = ShapeletTreeClassifier(
        random_state=1,
        strategy="random",
        metric="scaled_euclidean",
        n_shapelets=5,
        coverage_probability=0.2,
        variability=3,
    )
    f.fit(X_train, y_train)
    actual_apply = f.apply(X_test)
    # fmt:off
    expected_apply = np.array([
        2, 5, 8, 2, 7, 7, 2, 8, 5, 2, 2, 2, 8, 2, 2, 2, 5, 5, 5, 8, 2, 5,
        8, 2, 7, 5, 2, 2, 5, 8, 5, 5, 5, 5, 7, 2, 2, 5, 7, 8, 5, 5, 5, 5,
        5, 2, 3, 5, 2, 5, 5, 8, 2, 5, 5, 2, 8, 5, 2, 7, 7, 8, 2, 2, 2, 2,
        2, 5, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 7, 7, 8, 5, 5, 5, 2, 5,
        2, 8, 7, 8, 8, 5, 2, 2, 5, 5, 2, 2, 3, 7, 5, 5, 5, 7, 5, 5, 2, 8,
        5, 5, 2, 5, 7, 2, 5, 2, 7, 5, 7, 5, 5, 5, 7, 5, 7, 2, 8, 5, 5, 5,
        2, 2, 2, 3, 7, 5, 2, 2, 5, 7, 2, 8, 2, 5, 7, 5, 5, 7])
    # fmt: on
    assert_array_equal(actual_apply, expected_apply)


@pytest.mark.parametrize(
    "clf",
    [
        ShapeletTreeClassifier(strategy="random", random_state=1),
        ExtraShapeletTreeClassifier(random_state=1),
        ShapeletTreeClassifier(impurity_equality_tolerance=0.0, random_state=1),
    ],
)
@pytest.mark.parametrize("metric", ["euclidean"])
def test_shapelet_tree_benchmark(benchmark, clf, metric):
    X, y = load_gun_point()
    clf.set_params(metric=metric)

    def f(X, y):
        clf.fit(X, y)
        return clf.score(X, y)

    benchmark(f, X, y)
