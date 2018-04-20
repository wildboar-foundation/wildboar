import numpy as np
import time

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from pypf.tree import ShapeletTreeClassifier
from pypf._utils import print_tree


def testit():
    train = np.loadtxt("synthetic_control_TRAIN")
    test = np.loadtxt("synthetic_control_TEST")

    y = train[:, 0].astype(np.intp)
    x = train[:, 1:].astype(np.float64)
    tree = ShapeletTreeClassifier(n_shapelets=100, scale=True, max_depth=None)
    tree.fit(x, y)


if __name__ == "__main__":

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
    x = np.hstack([x, x]).reshape(-1, 2, x.shape[-1])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    random_state = np.random.RandomState(123)
    order = np.arange(10)
    random_state.shuffle(order)

    x = x[order, :]
    y = y[order]

    print(x)
    print(y)

    tree = ShapeletTreeClassifier(random_state=10, scale=True)
    tree.fit(x, y, sample_weight=np.ones(x.shape[0]) / x.shape[0])
    print_tree(tree.root_node_)
    print(tree.score(x, y))

    train = np.loadtxt("data/synthetic_control_TRAIN", delimiter=",")
    test = np.loadtxt("data/synthetic_control_TEST", delimiter=",")

    y = train[:, 0].astype(str)
    x = train[:, 1:].astype(np.float64)
    i = np.arange(x.shape[0])

    np.random.shuffle(i)

    x_test = test[:, 1:].astype(np.float64)
    y_test = test[:, 0].astype(str)

    tree = ShapeletTreeClassifier(
        n_shapelets=1,
        scale=False,
        max_depth=None,
        min_shapelet_size=1,
        max_shapelet_size=1)

    bag = BaggingClassifier(
        base_estimator=tree,
        bootstrap=True,
        n_jobs=8,
        n_estimators=100,
        random_state=100,
    )

    c = time.time()
    bag.fit(x, y)
    print(bag.classes_)
    print("acc:", bag.score(x_test, y_test))
    # score = cross_val_score(
    #     bag, np.vstack([x, x_test]), np.hstack([y, y_test]), cv=10)
    # print(np.mean(score), np.std(score))
    print(round(time.time() - c) * 1000)
