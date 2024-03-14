import numpy as np
import pytest
from wildboar.datasets import load_gun_point


@pytest.fixture(scope="session")
def gun_point():
    return load_gun_point(merge_train_test=False)


@pytest.fixture(scope="session")
def X(gun_point):
    return np.concatenate(gun_point[:2], axis=0)


@pytest.fixture(scope="session")
def X_train(gun_point):
    return gun_point[0]


@pytest.fixture(scope="session")
def X_test(gun_point):
    return gun_point[1]


@pytest.fixture(scope="session")
def y_train(gun_point):
    return gun_point[2]


@pytest.fixture(scope="session")
def y_test(gun_point):
    return gun_point[3]


@pytest.fixture(scope="session")
def y(gun_point):
    return np.concatenate(gun_point[2:], axis=None)


pytest.register_assert_rewrite("wildboar.utils._testing")
