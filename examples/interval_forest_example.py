import numpy as np

from wildboar.datasets import load_dataset
from wildboar.ensemble import IntervalForestClassifier

x_train, x_test, y_train, y_test = load_dataset("GunPoint", merge_train_test=False)

f = IntervalForestClassifier(
    n_interval=3,
    n_estimators=100,
    intervals="random",
    summarizer=[
        np.sum,
        np.mean,
        np.median,
        np.std,
        np.var,
        np.max,
        np.min,
        np.argmax,
        np.argmin,
    ],
    min_size=0.1,
    max_size=0.2,
    random_state=123,
)
f.fit(x_train, y_train)

print(f.score(x_test, y_test))
