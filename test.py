import numpy as np
import time

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

from pypf.tree import PfTree
from pypf._utils import print_tree

# from pypf._distribution import get_class_distribution
# from pypf._impurity import safe_info

# x = np.array([0, 1], dtype=np.float64)
# y = np.array([0.5, 0.5], dtype=np.float64)

x = [
    [0, 0, 1, 10, 1],
    [0, 0, 1, 10, 1],
    [0, 1, 9, 1, 0],
    [1, 9, 1, 0, 0],
    [0, 1, 9, 1, 0],
    [0, 1, 2, 3, 4],
    [1, 2, 3, 0, 0],
    [0, 0, 0, 1, 2],
    [0, 0, -1, 0, 1],
    [1, 2, 3, 0, 1],
]
x = np.array(x, dtype=np.float64)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

random_state = np.random.RandomState(123)
order = np.arange(10)
random_state.shuffle(order)

x = x[order, :]
y = y[order]

print(x)
print(y)

tree = PfTree(random_state=10)
tree.fit(x, y)

print_tree(tree.tree)
train = np.loadtxt("synthetic_control_TRAIN")
test = np.loadtxt("synthetic_control_TEST")

y = train[:, 0].astype(np.intp)
x = train[:, 1:].astype(np.float64)

x_test = test[:, 1:].astype(np.float64)
y_test = test[:, 0].astype(np.intp)

tree = PfTree(n_shapelets=100)


def max_depth(node, depth, max_d):
    if node.is_leaf:
        return max(depth, max_d)
    l_d = max_depth(node.left, depth + 1, max_d)
    r_d = max_depth(node.right, depth + 1, max_d)
    return max(l_d, max(r_d, max_d))


bag = BaggingClassifier(
    base_estimator=tree,
    bootstrap=True,
    n_jobs=16,
    n_estimators=100,
    random_state=100)
# print(np.vstack([x, x_test]).shape)
# print(np.hstack([y, y_test]).shape)

c = time.time()
bag.fit(x, y)
print(bag.score(x_test, y_test))
# score = cross_val_score(
#    bag, np.vstack([x, x_test]), np.hstack([y, y_test]), cv=10)
# print(score)
print(round(time.time() - c) * 1000)
