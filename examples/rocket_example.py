from wildboar.datasets import load_dataset
from wildboar.linear_model import RocketClassifier, RandomShapeletClassifier
from wildboar.tree._tree import RocketTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = load_dataset("FaceAll", merge_train_test=False)
f = BaggingClassifier(RocketTreeClassifier(n_kernels=10))
f.fit(x_train, y_train)
print(f.score(x_test, y_test))

f = RocketClassifier(n_kernels=10000)
f.fit(x_train, y_train)
print(f.score(x_test, y_test))

f = RandomShapeletClassifier(n_shapelets=10000)
f.fit(x_train, y_train)
print(f.score(x_test, y_test))