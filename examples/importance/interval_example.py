import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

from wildboar.datasets import load_dataset
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.explain import IntervalImportance

x, y = load_dataset("LargeKitchenAppliances")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=123
)

f = ShapeletForestClassifier(
    n_shapelets=1,
    n_estimators=100,
    n_jobs=-1,
    metric="scaled_euclidean",
    random_state=123,
)
f.fit(x_train, y_train)

i = IntervalImportance(
    n_interval=10,
    scoring="accuracy",
    domain="time",
    verbose=True,
    random_state=123,
)
i.fit(f, x_test, y_test)
ax = i.plot(
    x_test,
    y_test,
    top_k=0.2,
    n_samples=0.1,
)
plt.show()
