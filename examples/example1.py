import numpy as np
import time

from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble.weight_boosting import AdaBoostClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeClassifier

from wildboar.tree import ShapeletTreeClassifier
from wildboar.ensemble import ShapeletForestClassifier, ExtraShapeletTreesClassifier
from wildboar._utils import print_tree


def testit():
    train = np.loadtxt("synthetic_control_TRAIN")
    test = np.loadtxt("synthetic_control_TEST")

    y = train[:, 0].astype(np.intp)
    x = train[:, 1:].astype(np.float64)
    tree = ShapeletTreeClassifier(n_shapelets=100, max_depth=None)
    tree.fit(x, y)
    tree.score(test[:, 0], train[:, 1:])


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

    tree = ShapeletTreeClassifier(random_state=10, metric="euclidean")
    tree.fit(x, y)
    print("OOO")
    print(tree.tree_.node_count)
    print(tree.tree_.value)
    print(tree.tree_.shapelet)
    print(tree.tree_.n_node_samples)
    print(tree.tree_.n_weighted_node_samples)
    print(tree.tree_.left)
    print(tree.tree_.right)
    f = tree.tree_.apply(x)
    print(f)
    print(tree.classes_[np.argmax(np.take(tree.tree_.value, f, axis=0), axis=1)])
    # print("Score")
    # print(tree.score(x, y))
    # print("score_done")
    #
    train = np.loadtxt("data/synthetic_control_TRAIN", delimiter=",")
    test = np.loadtxt("data/synthetic_control_TEST", delimiter=",")
    #
    y = train[:, 0].astype(np.intp)
    x = train[:, 1:].astype(np.float64)
    i = np.arange(x.shape[0])

    np.random.shuffle(i)
    x_test = test[:, 1:].astype(np.float64)
    y_test = test[:, 0].astype(np.intp)

    f = ShapeletTreeClassifier(
        n_shapelets=5, metric="scaled_euclidean", metric_params={"r": 0.1}, random_state=112)
    #c = time.time()
    f.fit(x, y)
    print(f.tree_.n_node_samples)
    print(f.predict_proba(x_test))
    print(f.predict(x_test))
    print(f.score(x_test, y_test))

    import pickle

    pick = pickle.dumps(f)

    fx = pickle.loads(pick)
    print(fx.score(x_test, y_test))

    p = fx.decision_path(x_test)
    print(np.sum(p, axis=0))
    print(np.sum(p, axis=1))

    rf = ShapeletForestClassifier(n_jobs=16, metric="scaled_euclidean", n_estimators=100, oob_score=True, random_state=123)
    rf.fit(x, y)
    print(rf.score(x_test, y_test))
    erf = ExtraShapeletTreesClassifier(n_jobs=16, metric="euclidean", n_estimators=1000, random_state=123)
    erf.fit(x, y)
    print(erf.score(x_test, y_test))
    print(rf.estimators_[0].tree_.max_depth)
    print(erf.estimators_[0].tree_.max_depth)
    #print(f.classes_)
    # print(f.tree_.n_node_samples)
    # print("acc:", f.score(x_test, y_test))
    #print(round(time.time() - c) * 1000)
