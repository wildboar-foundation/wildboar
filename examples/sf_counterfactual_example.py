import numpy as np

from wildboar.ensemble import ShapeletForestClassifier
from wildboar.datasets import load_dataset
from wildboar.explain.counterfactual import counterfactual

x_train, x_test, y_train, y_test = load_dataset(
    "GunPoint", repository="wildboar/ucr", merge_train_test=False
)

clf = ShapeletForestClassifier(metric="euclidean", random_state=1, n_estimators=100)
clf.fit(x_train, y_train)

x_test = x_test[y_test == clf.classes_[0]]
y_tran = np.full(x_test.shape[0], clf.classes_[1])

x_counter, success, scores = counterfactual(
    clf, x_test, y_tran, random_state=123, scoring="euclidean"
)

print(scores[success])
print(np.sum(success) / success.shape[0])
print(np.sum(clf.predict(x_counter[success]) == y_tran[success]) / np.sum(success))
