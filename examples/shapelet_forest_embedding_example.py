import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline

from wildboar.datasets import load_dataset
from wildboar.ensemble import ShapeletForestEmbedding

random_state = 1234
x, y = load_dataset("GunPoint")
print(x.shape)

pipe = make_pipeline(
    ShapeletForestEmbedding(
        n_shapelets=1,
        min_shapelet_size=0,
        max_shapelet_size=1,
        metric="scaled_euclidean",
        sparse_output=True,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    ),
    LogisticRegression(solver="newton-cg", random_state=random_state),
)
cv = cross_validate(pipe, x, y, cv=5, scoring="accuracy", n_jobs=1)
print(1 - np.mean(cv["test_score"]))
