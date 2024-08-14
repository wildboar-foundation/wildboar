import numpy as np
import pytest
from wildboar.datasets import load_gun_point
from wildboar.transform import IntervalTransform


@pytest.mark.parametrize("intervals", ["dyadic", "fixed", "random"])
@pytest.mark.parametrize(
    "summarizer", ["quant", "mean_var_slope", "variance", "mean", "catch22"]
)
def test_interval_fit_transform(intervals, summarizer):
    X, y = load_gun_point()
    f = IntervalTransform(intervals=intervals, summarizer=summarizer, random_state=1)
    np.testing.assert_equal(f.fit_transform(X), f.fit(X).transform(X))


def test_dyadic_mean_var_slope():
    X, y = load_gun_point()
    f = IntervalTransform(intervals="dyadic")
    f.fit(X)

    # fmt: off
    expected = np.array([[
        -3.67428312e-04, -6.42234933e-01,  1.26615702e-05,
        -3.46267585e-04, -6.59057951e-01,  2.18781490e-05,
        -1.33097967e-03, -6.61435515e-01,  1.23188524e-06,
        2.71578630e-04, -6.56622553e-01
    ]])
    # fmt: on

    np.testing.assert_almost_equal(f.transform(X)[:1, 44:55], expected)
