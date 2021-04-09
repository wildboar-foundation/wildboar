import numpy as np
from wildboar.datasets import load_dataset
from wildboar.linear_model import (
    RocketClassifier,
    RandomShapeletClassifier,
)
from wildboar.tree._tree import RocketTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = load_dataset(
#    "FaceAll", repository="wildboar/ucr:no-missing", merge_train_test=False
# )
x, y = load_dataset("FaceAll", merge_train_test=True, preprocess="standardize")
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)
print(np.mean(x_train, axis=1))


# f = BaggingClassifier(RocketTreeClassifier(n_kernels=1000))
# f.fit(x_train, y_train)
# print(f.score(x_test, y_test))

f = RocketClassifier(
    n_kernels=10000,
    normalize_prob=0.5,
    bias_prob=0.5,
    sampling="uniform",
    sampling_params={"lower": -1, "upper": 1},
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
