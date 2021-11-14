import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

from wildboar.datasets import load_dataset
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.explain.importance import IntervalPermutationImportance

x, y = load_dataset("TwoLeadECG")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=123
)

f = ShapeletForestClassifier(
    n_shapelets=100,
    n_estimators=100,
    n_jobs=-1,
    metric="scaled_euclidean",
    random_state=123,
)
f.fit(x_train, y_train)

i = IntervalPermutationImportance(
    n_interval=20,
    scoring=["accuracy", "roc_auc"],
    verbose=True,
    random_state=123,
)
i.fit(f, x_test, y_test)

fig, ax = plt.subplots(nrows=2)
ax[0].plot(x_test[0])
i.plot(ax=ax[1], score="roc_auc")
plt.show()
