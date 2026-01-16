import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from wildboar.distance._manifold import MDS


@pytest.fixture
def sample_data():
    return np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])


def test_mds_default(sample_data):
    mds = MDS()
    transformed_data = mds.fit_transform(sample_data)
    assert transformed_data.shape == (3, 2)


def test_mds_nonmetric(sample_data):
    mds = MDS(metric_mds=False)
    transformed_data = mds.fit_transform(sample_data)
    assert transformed_data.shape == (3, 2)


def test_mds_custom_params(sample_data):
    mds = MDS(n_components=3, max_iter=100, eps=1e-5)
    transformed_data = mds.fit_transform(sample_data)
    assert transformed_data.shape == (3, 3)


def test_mds_invalid_dissimilarity():
    with pytest.raises(ValueError):
        mds = MDS(metric="invalid_metric")
        mds.fit_transform(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_mds_parallel_execution(sample_data):
    mds = MDS(n_jobs=2)
    transformed_data = mds.fit_transform(sample_data)
    assert transformed_data.shape == (3, 2)
