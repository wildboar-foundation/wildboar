import matplotlib.pylab as plt
import numpy as np

from wildboar.datasets import load_dataset
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.explain.counterfactual import PrototypeCounterfactual

x_train, x_test, y_train, y_test = load_dataset(
    "TwoLeadECG", repository="wildboar/ucr", merge_train_test=False
)
x_test_original = x_test
y_test_original = y_test

# clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
clf = ShapeletForestClassifier(
    n_estimators=100, metric="scaled_euclidean", random_state=10
)
# clf = RandomForestClassifier()
clf.fit(x_train, y_train)

cf = PrototypeCounterfactual(
    x_train,
    y_train,
    metric="dtw",
    metric_params={"r": 0.1},
    method="nearest_shapelet",
    method_params={"min_shapelet_size": 0.1, "max_shapelet_size": 0.2},
    target=0.70,
    step_size=1,
    max_iter=500,
    n_prototypes=10,
    random_state=3,
)
cf.fit(clf)

y_pred = clf.predict(x_test)
class_ = clf.classes_[1]
print("Class: %s" % class_)
print("Pred: %r" % y_pred)
print(" - where %r " % (y_pred != class_).nonzero())
x_test = x_test[y_test != class_][:10]
y_test = y_test[y_test != class_][:10]


x_counterfactual, success = cf.transform(
    x_test, np.broadcast_to(class_, x_test.shape[0])
)


print(clf.predict_proba(x_counterfactual))
print(y_test)
x_test = x_test[success]
x_counterfactual = x_counterfactual[success]

fig, ax = plt.subplots(nrows=3)
ax[0].plot(x_counterfactual[4], c="red")
ax[0].plot(x_test[4], c="blue")
ax[1].plot(x_counterfactual[1], c="red")
ax[1].plot(x_test[1], c="blue")
ax[1].legend(["x'", "x"])
# ax[2].plot(np.mean(x_counterfactual, axis=0), "r--")
ax[2].plot(x_test_original[0], "b--")
ax[2].plot(x_test_original[1], "g--")
plt.show()
