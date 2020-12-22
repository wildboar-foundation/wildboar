import numpy as np
from wildboar.datasets import load_dataset
from wildboar.explain.counterfactual import counterfactuals
from sklearn.neighbors import KNeighborsClassifier

x_train, x_test, y_train, y_test = load_dataset(
    "GunPoint", repository="wildboar/ucr", merge_train_test=False
)

clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
clf.fit(x_train, y_train)

x_counter, success, scores = counterfactuals(
    clf, x_test, y_test[::-1], random_state=123, scoring="euclidean"
)

print(scores)
print(np.sum(success))
print(np.sum(clf.predict(x_counter) == y_test[::-1]) / y_test.shape[0])
