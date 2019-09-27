from wildboar._tree_builder import RegressionShapeletTreeBuilder
from wildboar._tree_builder import RegressionShapeletTreePredictor
from wildboar._utils import print_tree
from wildboar.distance import DISTANCE_MEASURE
from sklearn.utils import check_random_state

from wildboar.tree import ShapeletTreeRegressor
import numpy as np

train = np.loadtxt("data/synthetic_control_TRAIN", delimiter=",")
test = np.loadtxt("data/synthetic_control_TEST", delimiter=",")

train = np.vstack([train, test])

y = train[:, 0].astype(np.float64)
random_state = check_random_state(123)
# for k in np.unique(y):
#     y[y == k] = random_state.randn(y[y == k].shape[0]) + k
y = np.ascontiguousarray(y)
x = np.ascontiguousarray(train[:, 1:].astype(np.float64))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=random_state)

random_state = check_random_state(123)
# tree_builder = RegressionShapeletTreeBuilder(
#     1,
#     2,
#     60,
#     1000,
#     DISTANCE_MEASURE["euclidean"](150),
#     x,
#     y,
#     None,
#     random_state,
# )

# tree = tree_builder.build_tree()
# print_tree(tree)

# pred = RegressionShapeletTreePredictor(x, DISTANCE_MEASURE["euclidean"](150))
# print(np.linalg.norm(pred.predict(tree) - y))

r = ShapeletTreeRegressor(
    n_shapelets=1,
    metric="euclidean",
    max_depth=None,
    min_samples_split=2,
    random_state=check_random_state(123))

from sklearn.ensemble import BaggingRegressor
from wildboar.ensemble import ShapeletForestRegressor

b = ShapeletForestRegressor(
    random_state=check_random_state(123),
    n_jobs=8,
    bootstrap=False,
    metric="scaled_dtw",
    metric_params={"r": 0.2})
#BaggingRegressor(r, bootstrap=False, n_jobs=4, n_estimators=100)

b.fit(x_train, y_train)
print(b.predict(x_test))
print(b.score(x_test, y_test))

from sklearn.neighbors import KNeighborsRegressor

print(
    KNeighborsRegressor(n_neighbors=1).fit(x_train, y_train).score(
        x_test, y_test))
