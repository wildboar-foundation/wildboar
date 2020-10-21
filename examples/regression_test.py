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

print(x_train.shape)

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
    n_shapelets=100,
    metric="scaled_euclidean",
    max_depth=None,
    min_samples_split=2,
    random_state=check_random_state(123))

from sklearn.ensemble import BaggingRegressor
from wildboar.ensemble import ShapeletForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
c = ExtraTreesRegressor(max_depth=int(np.ceil(np.log2(max(x_train.shape[0], 2)))),)
b = ShapeletForestRegressor(
    random_state=123,
    n_estimators=100,
    max_depth=30,
    n_jobs=16,
    n_shapelets=10,
    bootstrap=False,
    metric="scaled_euclidean",
    metric_params={"r": 0.2})

b.fit(x_train, y_train)#np.random.uniform(size=x_train.shape[0]))
print(b.score(x_test, y_test))
print(b.estimators_[0].tree_.max_depth)
#print(r.tree_.n_node_samples)
#print(r.tree_.apply(x_train))
#print(r.predict(x_test))
#print(y_test)
#print(r.score(x_test, y_test))
#b.fit(x_train, np.random.uniform(size=x_train.shape[0]))
# print(b.bagging_regressor_.estimators_[0].apply(x_test))
#print(np.ravel(b.estimators_[0].decision_path(x_test).sum(axis=1)))
#print(np.ravel(c.estimators_[0].decision_path(x_test).sum(axis=1)))

#print(c.estimators_[0].tree_.n_node_samples[c.estimators_[0].apply(x_test)])
#leaves = b.estimators_[0].apply(x_test)
#print(b.estimators_[0].n_node_samples[leaves])
#is_inlier = np.ones(x_test.shape[0])
#is_inlier[score < 0] = -1
#print(is_inlier)
#print(y_test)
#print(b.score(x_test, y_test))

#print(np.where(is_inlier == -1))
