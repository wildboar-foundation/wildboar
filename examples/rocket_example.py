import numpy as np
from sklearn.model_selection import train_test_split

from wildboar.datasets import load_dataset
from wildboar.linear_model import RandomShapeletClassifier, RocketClassifier

x, y = load_dataset("ItalyPowerDemand", merge_train_test=True, preprocess="standardize")
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)
print(np.mean(x_train, axis=1))


f = RocketClassifier(
    n_kernels=10000,
    sampling="normal",
    sampling_params={"mean": 0, "scale": 1},
    n_jobs=16,
    random_state=123,
    alphas=np.logspace(-3, 3, 10),
    normalize=True,
)
f.fit(x_train, y_train)
print(f.score(x_test, y_test))


f = RandomShapeletClassifier(
    n_shapelets=10000,
    n_jobs=16,
    metric="scaled_euclidean",
    random_state=123,
    alphas=np.logspace(-3, 3, 10),
    normalize=True,
)
f.fit(x_train, y_train)
print(f.score(x_test, y_test))
