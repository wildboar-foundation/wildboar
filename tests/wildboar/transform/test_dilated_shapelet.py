import numpy as np
from wildboar.datasets import load_gun_point
from wildboar.transform import (
    CastorTransform,
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
        0.29630489,  44.        , 133.,           0.35263642, 113.,
        1.        ,   0.52851197,  59.,          42.        ,
    ]])
    # fmt: on

    np.testing.assert_almost_equal(f.transform(X)[:1], expected)


def test_dilated_shapelet_supervised_transform():
    X, y = load_gun_point()
    f = DilatedShapeletTransform(random_state=1, n_shapelets=3, normalize_prob=0.5)
    f.fit(X, y)

    # fmt: off
    expected = np.array([[
         0.29630489,  44.        ,  38.        ,   0.18232488,
        97.        ,  91.        ,   1.58282394, 132.        ,
       123.
    ]])
    # fmt: on
    np.testing.assert_almost_equal(f.transform(X)[:1], expected)


def test_competing_dilated_shapelet_unsupervised_transform():
    X, y = load_gun_point()
    f = CastorTransform(random_state=1, n_groups=1, normalize_prob=0.5)
    f.fit(X)

    # fmt: off
    expected = np.array([[
        9.19112402 ,  8.         ,  2.         , 58.99305918 , 36.,
        8.         , 79.39677436 , 42.         ,  4.         , 33.19625078,
        17.        ,   5.        ,  30.44073231,  23.        ,  11.,
        44.18639532,  20.        ,   6.        ,  12.3379486 ,   3.,
        8.         ,  2.37855733 ,  1.         ,  8.         , 13.23868117,
        12.        ,   9.        ,  63.46859221,  34.        ,  14.,
        100.640464 ,   57.       ,   11.       ,    0.       ,    2.,
        18.        ,  10.50423492,   5.        ,  12.        ,  51.73939936,
        25.        ,  23.        ,   3.56259367,   9.        ,   8.,
        32.27444158,   6.        ,   0.        ,   3.88664852,   6.,
        8.         , 46.08216362 , 23.         ,  4.         , 23.79138901,
        2.         ,  0.         , 58.71352008 , 37.         , 11.,
        5.27236614 , 11.         , 11.         , 52.65005572 , 14.,
        10.        ,  14.75470266,  13.        ,   8.        , 111.76788013,
        44.        ,  12.        ,   3.1576424 ,  31.        ,   0.,
        98.08326993,  18.        ,   0.        , 176.82351976,  34.,
        3.         , 53.73938124 , 16.         ,  8.         , 89.6030808,
        41.        ,   0.        ,   3.55354454,   0.        ,   8.,
        5.96255535 ,  2.         ,  9.         ,  5.63377047 ,  8.,
        10.        ,
    ]])
    # fmt: on
    np.testing.assert_almost_equal(f.transform(X[:1])[0], expected[0])


def test_competing_dilated_shapelet_supervised_transform():
    X, y = load_gun_point()
    f = CastorTransform(random_state=1, n_groups=1, normalize_prob=0.5)
    f.fit(X, y)

    # fmt: off
    expected = np.array([[
         2.79517716,   6.        ,   6.        ,   3.10118429,
         3.        ,  12.        ,  50.78637862,  25.        ,
         3.        ,  27.5667441 ,  29.        ,  25.        ,
         8.67376659,   9.        ,   7.        ,  79.46492513,
        53.        ,   9.        ,  44.34552163,  22.        ,
        33.        ,   9.14099599,   3.        ,  14.        ,
        45.05797145,  31.        ,  11.        ,  22.03220736,
        15.        ,  10.        ,  48.38316964,  24.        ,
         9.        ,   6.37536621,   5.        ,   3.        ,
        12.07079038,  28.        ,  11.        ,  11.81626848,
        16.        ,  10.        ,   6.72615863,   9.        ,
        10.        ,  29.06713623,  22.        ,   6.        ,
         4.14407542,  11.        ,   9.        ,   6.79998137,
        20.        ,   9.        ,  24.8230681 ,  21.        ,
         7.        ,   5.63273987,   4.        ,   5.        ,
       117.41249373,  44.        ,  10.        ,   7.4643829 ,
        17.        ,  12.        ,   7.92905134,   7.        ,
         9.        ,  60.81964633,  26.        ,  11.        ,
        22.55192706,  31.        ,   0.        , 150.05419053,
        44.        ,   0.        ,  68.2379208 ,  16.        ,
        12.        , 129.33618657,  20.        ,  23.        ,
         0.        ,  11.        ,  12.        ,   8.38214917,
         0.        ,  27.        ,   8.84835479,  10.        ,
        18.        ,  15.6391129 ,  18.        ,  22.
    ]])
    # fmt: on
    actual = f.transform(X[:1])
    np.testing.assert_almost_equal(actual[0], expected[0])
