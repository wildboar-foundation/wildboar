import catch22
import pandas as pd

from wildboar.datasets import load_dataset
from wildboar.embed import IntervalEmbedding

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

i = IntervalEmbedding(
    n_interval=1,
    summarizer=summarizers,
    random_state=123,
)
x_t = i.fit_transform(x_train[[0, 1, 13]])
df_expected = pd.DataFrame(x_t, columns=names)

i = IntervalEmbedding(
    n_interval=1,
    summarizer="catch22",
    random_state=123,
)
x_t = i.fit_transform(x_train[[0, 1, 13]])
df_actual = pd.DataFrame(x_t, columns=names)

with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.width", None
):
    print(df_expected[names[0:22]])
    print(df_actual[names[0:22]])


#
# pipe = make_pipeline(
#     IntervalEmbedding(
#         n_interval=1,
#         intervals="fixed",
#         summarizer=summarizers,
#     ),
#     RandomForestClassifier(),
# )
# pipe.fit(x_train, y_train)
# print(pipe.score(x_test, y_test))
