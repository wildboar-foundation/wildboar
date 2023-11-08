import numpy as np
from wildboar.datasets import load_gun_point
from wildboar.transform import (
    CompetingDilatedShapeletTransform,
    DilatedShapeletTransform,
)


def test_dilated_shapelet_fit_transform():
    X, y = load_gun_point()
    f = DilatedShapeletTransform(n_shapelets=100, random_state=1)
    np.testing.assert_equal(f.fit_transform(X), f.fit(X).transform(X))


def test_dilated_shapelet_unsupervised_transform():
    X, y = load_gun_point()
    f = DilatedShapeletTransform(random_state=1, n_shapelets=3, normalize_prob=0.5)
    f.fit(X)

    # fmt: off
    expected = np.array([[
        0.29630489,  44.        , 139.        ,   0.35263642,
      113.        , 100.        ,   0.52851197,  59.        ,
       45.
    ]])
    # fmt: on

    np.testing.assert_almost_equal(f.transform(X)[:1], expected)


def test_dilated_shapelet_supervised_transform():
    X, y = load_gun_point()
    f = DilatedShapeletTransform(random_state=1, n_shapelets=3, normalize_prob=0.5)
    f.fit(X, y)

    # fmt: off
    expected = np.array([[
        0.29630489,  44.,          39.,           0.18232488,  97.,
       62.,           1.13058855, 132.,         107.,
    ]])
    # fmt: on
    np.testing.assert_almost_equal(f.transform(X)[:1], expected)


def test_competing_dilated_shapelet_unsupervised_transform():
    X, y = load_gun_point()
    f = CompetingDilatedShapeletTransform(
        random_state=1, n_groups=1, normalize_prob=0.5
    )
    f.fit(X)

    # fmt: off
    expected = np.array([[
          9.19112402,   8.        ,   2.        ,  56.25746282,
         36.        ,   8.        ,  77.75541586,  42.        ,
          4.        ,  33.19625078,  17.        ,   5.        ,
         28.88952187,  23.        ,  11.        ,  41.33853811,
         20.        ,   6.        ,  12.07178431,   3.        ,
          9.        ,   2.37855733,   1.        ,   7.        ,
         23.44982993,  14.        ,  12.        ,  49.34049391,
         31.        ,  28.        ,   4.50553843,   4.        ,
          8.        ,  13.91944223,  23.        ,  15.        ,
          3.52573634,   8.        ,  10.        ,  36.19979131,
         18.        ,  17.        ,  68.78082406,  38.        ,
          6.        ,  10.59844513,  14.        ,   7.        ,
          9.94123782,  19.        ,  11.        ,  30.88849659,
          5.        ,  21.        ,   4.11572748,   9.        ,
          0.        ,   0.        ,  12.        ,   8.        ,
         37.53750457,  23.        ,   0.        ,  43.73596501,
         27.        ,  14.        ,  64.78862123,  12.        ,
         19.        , 107.88550005,  43.        ,   9.        ,
          7.36459551,   0.        ,  10.        ,  36.63922329,
         28.        ,   0.        , 103.07174581,  46.        ,
         12.        ,  62.54022822,  51.        ,   0.        ,
        131.2093716 ,  18.        ,   0.        ,   0.        ,
          0.        ,   0.        ,   5.34456813,   7.        ,
          6.        ,  13.3015121 ,   0.        ,   0.
    ]])
    # fmt: on
    np.testing.assert_almost_equal(f.transform(X[:1]), expected)


def test_competing_dilated_shapelet_supervised_transform():
    X, y = load_gun_point()
    f = CompetingDilatedShapeletTransform(
        random_state=1, n_groups=1, normalize_prob=0.5
    )
    f.fit(X, y)

    # fmt: off
    expected = np.array([[
          2.79517716,   6.        ,   4.        ,   3.10118429,
          3.        ,   5.        ,  50.78637862,  25.        ,
          3.        ,  26.95257546,  29.        ,  19.        ,
          8.39650457,   9.        ,   8.        ,  77.80454875,
         53.        ,  10.        ,  44.34552163,  22.        ,
          5.        ,   8.67074986,   3.        ,  12.        ,
          6.85908051,   8.        ,   5.        ,  44.98167442,
         23.        ,  17.        ,  13.91116268,   9.        ,
         12.        ,  21.12454944,  11.        ,  10.        ,
         33.89967888,  22.        ,  11.        ,   8.34849652,
          8.        ,   6.        ,  42.5773261 ,  47.        ,
          9.        ,  56.08096022,  22.        ,  16.        ,
          7.0090699 ,  14.        ,  10.        ,  18.93297367,
         13.        ,  17.        , 136.80222588,  48.        ,
         10.        ,   0.        ,   0.        ,  15.        ,
          8.59094942,  19.        ,   8.        ,   8.7827816 ,
         16.        ,  13.        ,   8.81969199,   8.        ,
         12.        ,  80.39272769,  32.        ,  13.        ,
          7.41532376,   9.        ,  15.        ,   0.        ,
         19.        ,   0.        ,   0.        ,   0.        ,
          0.        ,   9.24882026,  24.        ,  13.        ,
        155.62995768,  37.        ,  19.        ,  69.37210544,
         34.        ,  12.        ,  10.56858841,   0.        ,
         31.        , 107.75296436,  27.        ,  17.
    ]])
    # fmt: on
    actual = f.transform(X[:1])
    np.testing.assert_almost_equal(actual, expected)
