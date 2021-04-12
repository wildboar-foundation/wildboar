from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from wildboar.datasets import load_dataset
from wildboar.embed import RocketEmbedding

x_train, x_test, y_train, y_test = load_dataset("GunPoint", merge_train_test=False)


pipe = make_pipeline(RocketEmbedding(n_kernels=1000), RidgeClassifierCV())
pipe.fit(x_train, y_train)
print(pipe.score(x_test, y_test))
