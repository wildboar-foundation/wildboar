from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from wildboar.datasets import load_dataset
from wildboar.embed import RandomShapeletEmbedding

x_train, x_test, y_train, y_test = load_dataset("GunPoint", merge_train_test=False)


pipe = make_pipeline(
    RandomShapeletEmbedding(metric="scaled_euclidean"), RidgeClassifierCV()
)
pipe.fit(x_train, y_train)
print(pipe.score(x_test, y_test))
