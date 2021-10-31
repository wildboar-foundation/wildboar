from timeit import default_timer as timer

import catch22
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from wildboar.datasets import load_dataset
from wildboar.embed import IntervalEmbedding
from wildboar.tree import IntervalTreeClassifier

x, y = load_dataset("GunPoint")
x_train, x_test, y_train, y_test = load_dataset("GunPoint", merge_train_test=False)


def wrap(f):
    def w(x):
        return f(list(x))

    return w


summarizers = [
    wrap(f)
    for f in catch22.__dict__.values()
    if callable(f) and f.__name__ != "catch22_all"
]

names = [
    f.__name__
    for f in catch22.__dict__.values()
    if callable(f) and f.__name__ != "catch22_all"
]

selection = np.arange(x.shape[0])
i = IntervalEmbedding(
    n_interval=1,
    summarizer=summarizers,
    random_state=123,
)


start = timer()
x_t = i.fit_transform(x[selection])
end = timer()
print("non-native: ", end - start)

df_expected = pd.DataFrame(x_t, columns=names)

i = IntervalEmbedding(
    n_interval=1,
    summarizer="catch22",
    random_state=123,
)
start = timer()
x_t = i.fit_transform(x[selection])
end = timer()
print("native: ", end - start)
df_actual = pd.DataFrame(x_t, columns=names)

# df_expected["SB_BinaryStats_mean_longstretch1"] += 1
pd.testing.assert_frame_equal(df_expected, df_actual)
# with pd.option_context(
#     "display.max_rows", None, "display.max_columns", None, "display.width", None
# ):
#     #print(df_expected - df_actual)

pipe = make_pipeline(
    IntervalEmbedding(
        n_interval=1000,
        intervals="random",
        summarizer="catch22",
        min_size=0.1,
        max_size=0.2,
        n_jobs=-1,
    ),
    RidgeClassifierCV(),
)
# pipe.fit(x_train, y_train)
# print(pipe.score(x_test, y_test))

t = IntervalTreeClassifier(
    n_interval=100,
    intervals="random",
    min_size=0.3,
    max_size=0.4,
    summarizer="catch22",
)
t.fit(x_train, y_train)
print("Fit?")
print(t.score(x_test, y_test))
