from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from wildboar.datasets import load_dataset
from wildboar.embed import IntervalEmbedding

x_train, x_test, y_train, y_test = load_dataset("GunPoint", merge_train_test=False)

i = IntervalEmbedding(
    n_interval="sqrt",
    intervals="random",
    min_size=0.0,
    max_size=0.3,
    summarizer="auto",
    random_state=123,
)
i.fit(x_train)
print([(a, b) for _, (a, b, c) in i.embedding_.features])


pipe = make_pipeline(
    IntervalEmbedding(n_interval=50, summarizer="auto"), RidgeClassifierCV()
)
pipe.fit(x_train, y_train)
print(pipe.score(x_test, y_test))
