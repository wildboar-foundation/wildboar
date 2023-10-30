import numpy as np
from wildboar.datasets import load_gun_point
from wildboar.transform import DilatedShapeletTransform


def test_dilated_shapelet_transform_fit_transform():
    X, y = load_gun_point()
    f = DilatedShapeletTransform(n_shapelets=100, random_state=1)
    np.testing.assert_equal(f.fit_transform(X), f.fit(X).transform(X))


def test_hydra_transform():
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
