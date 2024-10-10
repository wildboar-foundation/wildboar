import numpy as np
import pytest

from wildboar.transform import MatrixProfileTransform


class TestMatrixProfileTransform:
    def test_initialization(self):
        transformer = MatrixProfileTransform()
        assert transformer.window == 0.1
        assert transformer.exclude is None
        assert transformer.n_jobs is None

    def test_fit_valid_data(self):
        transformer = MatrixProfileTransform(window=0.2)
        x = np.random.rand(10, 20)  # 10 samples, 20 timesteps
        transformer.fit(x)

    def test_transform_valid_data(self):
        transformer = MatrixProfileTransform(window=0.2)
        x = np.random.rand(10, 20)  # 10 samples, 20 timesteps
        transformer.fit(x)
        result = transformer.transform(x)
        assert result.shape == (10, 17)  # 10 samples, 4 timesteps after transformation

    def test_fit_window_too_large(self):
        transformer = MatrixProfileTransform(window=200)
        x = np.random.rand(10, 20)  # 10 samples, 20 timesteps
        with pytest.raises(ValueError):
            transformer.fit(x)

    def test_transform_3d_array(self):
        transformer = MatrixProfileTransform(window=0.2)
        x = np.random.rand(10, 3, 20)  # 10 samples, 3 dimensions, 20 timesteps
        transformer.fit(x)
        result = transformer.transform(x)
        assert result.shape == (
            10,
            3,
            17,
        )  # 10 samples, 3 dimensions, 4 timesteps after transformation

    def test_fit_invalid_data_type(self):
        transformer = MatrixProfileTransform(window=0.2)
        x = "invalid_data"
        with pytest.raises(ValueError):
            transformer.fit(x)

    def test_transform_invalid_data_type(self):
        transformer = MatrixProfileTransform(window=0.2)
        x = "invalid_data"
        with pytest.raises(ValueError):
            transformer.transform(x)
