import numpy as np
from sklearn.utils import check_random_state

train = np.loadtxt("data/synthetic_control_TRAIN", delimiter=",")
test = np.loadtxt("data/synthetic_control_TEST", delimiter=",")

train = np.vstack([train, test])

random = check_random_state(123)
y = train[:, 0].astype(np.float64)
one = (y == 1) | (y == 2) | (y == 3)
y[one] = random.randn(np.sum(one))
y[~one] = 10 + random.randn(np.sum(~one))

# for k in np.unique(y):
#     y[y == k] = random_state.randn(y[y == k].shape[0]) + k
y = np.ascontiguousarray(y)
x = np.ascontiguousarray(train[:, 1:].astype(np.float64))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=123
)

print(x_train.shape)

from wildboar.ensemble import ExtraShapeletTreesRegressor

b = ExtraShapeletTreesRegressor(
    random_state=123,
    n_estimators=100,
    n_jobs=16,
    min_shapelet_size=0,
    max_shapelet_size=1,
    bootstrap=False,
    metric="euclidean",
    metric_params={"r": 0.2},
)

b.fit(x_train, y_train)
print(b.score(x_test, y_test))
n_node_samples = b.estimators_[1].tree_.n_node_samples
left = b.estimators_[1].tree_.left
right = b.estimators_[1].tree_.right
value = b.estimators_[1].tree_.value
from sklearn.neighbors import KNeighborsRegressor

print(KNeighborsRegressor().fit(x_train, y_train).score(x_test, y_test))
