# Authors: Isak Samsten
# License: BSD 3 clause

import pytest
from numpy.testing import assert_almost_equal, assert_equal
from wildboar.datasets import load_dataset, load_gun_point
from wildboar.ensemble import ShapeletForestClassifier, ShapeletForestRegressor
from wildboar.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(
        ShapeletForestClassifier(n_shapelets=10, n_estimators=10),
        ignore=["check_sample_weights_invariance"],
    )
    check_estimator(
        ShapeletForestRegressor(n_shapelets=10, n_estimators=10),
        ignore=["check_sample_weights_invariance"],
    )


@pytest.mark.parametrize(
    "estimator, expected_estimator_params",
    [
        (
            ShapeletForestClassifier(),
            (
                "max_depth",
                "n_shapelets",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "min_shapelet_size",
                "max_shapelet_size",
                "alpha",
                "metric",
                "metric_params",
                "criterion",
            ),
        ),
        (
            ShapeletForestRegressor(),
            (
                "max_depth",
                "n_shapelets",
                "min_samples_split",
                "min_samples_leaf",
                "min_impurity_decrease",
                "min_shapelet_size",
                "max_shapelet_size",
                "alpha",
                "metric",
                "metric_params",
                "criterion",
            ),
        ),
    ],
)
def test_check_estimator_params(estimator, expected_estimator_params):
    assert set(estimator.estimator_params) == set(expected_estimator_params)


def test_shapelet_forest_classifier():
    x_train, x_test, y_train, y_test = load_dataset(
        "GunPoint", repository="wildboar/ucr-tiny", merge_train_test=False
    )
    clf = ShapeletForestClassifier(n_estimators=10, n_shapelets=10, random_state=1)
    clf.fit(x_train, y_train)
    branches = [
        (
            [1, 2, -1, 4, -1, 6, -1, 8, -1, -1, -1],
            [10, 3, -1, 5, -1, 7, -1, 9, -1, -1, -1],
        ),
        ([1, -1, 3, 4, -1, 6, -1, -1, -1], [2, -1, 8, 5, -1, 7, -1, -1, -1]),
        ([1, -1, 3, -1, 5, -1, -1], [2, -1, 4, -1, 6, -1, -1]),
        ([1, -1, 3, 4, -1, 6, -1, -1, -1], [2, -1, 8, 5, -1, 7, -1, -1, -1]),
        ([1, 2, -1, 4, -1, -1, 7, -1, -1], [6, 3, -1, 5, -1, -1, 8, -1, -1]),
        ([1, -1, 3, -1, 5, -1, -1], [2, -1, 4, -1, 6, -1, -1]),
        (
            [1, -1, 3, -1, 5, 6, 7, -1, -1, -1, -1],
            [2, -1, 4, -1, 10, 9, 8, -1, -1, -1, -1],
        ),
        ([1, -1, 3, 4, 5, -1, -1, -1, -1], [2, -1, 8, 7, 6, -1, -1, -1, -1]),
        ([1, 2, -1, 4, -1, -1, 7, -1, -1], [6, 3, -1, 5, -1, -1, 8, -1, -1]),
        ([1, 2, -1, 4, -1, -1, -1], [6, 3, -1, 5, -1, -1, -1]),
    ]

    thresholds = [
        (
            [
                1.1940153159491205,
                1.5668609643722844,
                0.7507155758809275,
                2.3512055788657005,
                4.192706964103458,
            ],
            [
                1.1940153159491205,
                1.5668609643722844,
                0.7507155758809275,
                2.3512055788657005,
                4.192706964103458,
            ],
        ),
        (
            [
                2.4685040043186115,
                1.741428808363988,
                0.294665870209171,
                8.08972721370202,
            ],
            [
                2.4685040043186115,
                1.741428808363988,
                0.294665870209171,
                8.08972721370202,
            ],
        ),
        (
            [5.196214979400872, 0.45335153612799833, 2.8946919237523785],
            [5.196214979400872, 0.45335153612799833, 2.8946919237523785],
        ),
        (
            [
                0.7885454744609631,
                1.2949878959699022,
                4.180099199070423,
                0.32665343218739884,
            ],
            [
                0.7885454744609631,
                1.2949878959699022,
                4.180099199070423,
                0.32665343218739884,
            ],
        ),
        (
            [
                2.784221792615396,
                1.1200408846363272,
                7.835009838927329,
                9.849133641222553,
            ],
            [
                2.784221792615396,
                1.1200408846363272,
                7.835009838927329,
                9.849133641222553,
            ],
        ),
        (
            [1.2903260623847297, 0.4526518197877716, 10.955500264160882],
            [1.2903260623847297, 0.4526518197877716, 10.955500264160882],
        ),
        (
            [
                0.8027463491464661,
                1.2232800197618816,
                2.157674858284418,
                0.2894374818511062,
                1.3423736488142453,
            ],
            [
                0.8027463491464661,
                1.2232800197618816,
                2.157674858284418,
                0.2894374818511062,
                1.3423736488142453,
            ],
        ),
        (
            [
                0.6168414952987998,
                2.5536055762458627,
                7.659014985104224,
                0.43821283355679563,
            ],
            [
                0.6168414952987998,
                2.5536055762458627,
                7.659014985104224,
                0.43821283355679563,
            ],
        ),
        (
            [
                2.955206215288551,
                0.605227644212164,
                8.862523891605228,
                0.13362980606936598,
            ],
            [
                2.955206215288551,
                0.605227644212164,
                8.862523891605228,
                0.13362980606936598,
            ],
        ),
        (
            [7.142302030513696, 1.0892908889283355, 7.873172873651155],
            [7.142302030513696, 1.0892908889283355, 7.873172873651155],
        ),
    ]

    for estimator, (left, right), (left_threshold, right_threshold) in zip(
        clf.estimators_, branches, thresholds
    ):
        assert_equal(left, estimator.tree_.left)
        assert_equal(right, estimator.tree_.right)
        assert_almost_equal(
            left_threshold, estimator.tree_.threshold[estimator.tree_.left > 0]
        )
        assert_almost_equal(
            right_threshold, estimator.tree_.threshold[estimator.tree_.right > 0]
        )


def test_fit_3d_n_dims_1():
    X, y = load_gun_point()
    X_3d = X.reshape(X.shape[0], 1, -1)
    clf0 = ShapeletForestClassifier(random_state=1)
    clf1 = ShapeletForestClassifier(random_state=1)
    clf0.fit(X_3d, y)
    clf1.fit(X, y)
    assert (clf0.predict(X_3d) == clf1.predict(X)).sum() == y.shape[0]
    assert (clf0.predict(X) == clf1.predict(X_3d)).sum() == y.shape[0]

    for b0, b1 in zip(clf0.estimators_, clf1.estimators_):
        assert_equal(b0.tree_.left[b0.tree_.left > 0], b1.tree_.left[b1.tree_.left > 0])
        assert_equal(
            b0.tree_.right[b0.tree_.right > 0], b1.tree_.right[b1.tree_.right > 0]
        )
