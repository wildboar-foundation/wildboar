import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from wildboar.datasets import load_dataset
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.explain.counterfactual import counterfactuals
from wildboar.linear_model import RocketClassifier

x, y = load_dataset("GunPoint")
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)

rocket = RocketClassifier(n_kernels=1000, n_jobs=-1, random_state=123)
rocket.fit(x_train, y_train)
print("Rocket score", rocket.score(x_test, y_test))

sf = ShapeletForestClassifier(n_shapelets=10, n_jobs=-1, random_state=123)
sf.fit(x_train, y_train)
print("Shapelet score", sf.score(x_test, y_test))

nearest = KNeighborsClassifier(n_jobs=-1)
nearest.fit(x_train, y_train)
print("Neighbors score", nearest.score(x_test, y_test))

x_test_cls = x_test[y_test == 1.0][:2]

for method in [rocket, sf, nearest]:
    print(method)
    x_counter, x_success = counterfactuals(
        method,
        x_test_cls,
        2.0,
        method="prototype",
        random_state=123,
        method_args={
            "method": "nearest_shapelet",
            "metric": "euclidean",
            "max_iter": 100,
            "step_size": 0.05,
            "n_prototypes": 5,
            "train_x": x_train,
            "train_y": y_train,
        },
    )

    print(x_success)
    print(method.predict(x_test_cls))
    print(method.predict(x_counter))
    plt.title(method)
    plt.plot(x_counter[1], "r")
    plt.plot(x_test_cls[1], "g")
    plt.show()
