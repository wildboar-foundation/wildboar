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
         10.47538881,  12.        ,  11.        ,  59.81251052,
         34.        ,  14.        ,  94.44716759,  57.        ,
         11.        ,   0.        ,   2.        ,  11.        ,
         10.03956407,   5.        ,  12.        ,  51.13186156,
         25.        ,  23.        ,   3.56259367,   9.        ,
          8.        ,  31.16533414,   6.        ,   0.        ,
          3.88664852,   6.        ,  14.        ,  42.9265914 ,
         23.        ,   4.        ,  23.79138901,   2.        ,
          0.        ,  54.39350151,  37.        ,  18.        ,
          5.27236614,  11.        ,  10.        ,  49.83281679,
         14.        ,  18.        ,  14.75470266,  13.        ,
          8.        ,  99.68841499,  44.        ,   8.        ,
          3.1576424 ,  31.        ,   0.        ,  79.87461999,
         18.        ,   0.        , 127.88509521,  34.        ,
          1.        ,  34.16113893,  16.        ,   8.        ,
         87.40451983,  41.        ,   0.        ,   3.55354454,
          0.        ,   8.        ,   5.96255535,   2.        ,
          9.        ,   5.63377047,   8.        ,  10.
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
         41.33778592,  31.        ,  16.        ,  20.40928765,
         15.        ,  12.        ,  48.38316964,  24.        ,
         10.        ,   6.37536621,   5.        ,   6.        ,
         12.07079038,  28.        ,  12.        ,  11.81626848,
         16.        ,  12.        ,   6.72615863,   9.        ,
          7.        ,  28.21047653,  22.        ,  14.        ,
          4.14407542,  11.        ,   9.        ,   6.79998137,
         20.        ,   9.        ,  24.00652585,  21.        ,
          8.        ,   5.63273987,   4.        ,   8.        ,
        109.33681946,  44.        ,  20.        ,   7.4643829 ,
         17.        ,  16.        ,   7.92905134,   7.        ,
         11.        ,  54.57099944,  26.        ,  11.        ,
         22.55192706,  31.        ,   0.        , 141.09763404,
         44.        ,  19.        ,  53.43507259,  16.        ,
          0.        ,  86.83099771,  20.        ,  16.        ,
          0.        ,  11.        ,  11.        ,   8.38214917,
          0.        ,   9.        ,   8.84835479,  10.        ,
         13.        ,  13.30142277,  18.        ,  11.
    ]])
    # fmt: on
    actual = f.transform(X[:1])
    np.testing.assert_almost_equal(actual, expected)
