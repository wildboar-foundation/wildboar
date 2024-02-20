from numpy.testing import assert_array_equal
from wildboar.datasets import load_ering
from wildboar.dimension_selection import (
    DistanceVarianceThreshold,
    SequentialDimensionSelector,
)
from wildboar.distance import KNeighborsClassifier


def test_sequential_dimension_selector():
    X, y = load_ering()
    clf = KNeighborsClassifier()
    sds = SequentialDimensionSelector(clf, n_dims=2)
    sds.fit(X, y)
    assert_array_equal(sds.get_dimensions(), [True, False, False, True])
    X_t = sds.transform(X)
    assert_array_equal(X_t, X[:, [True, False, False, True], :])


def test_variance_dimension_selector():
    X, y = load_ering()
    t = DistanceVarianceThreshold(threshold=9, sample=0.5, random_state=1)
    t.fit(X, y)
    assert_array_equal(t.get_dimensions(), [True, False, True, True])
