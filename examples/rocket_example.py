from wildboar.datasets import load_dataset
from wildboar.linear_model import RocketClassifier
from wildboar.tree._tree import RocketTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split


x, y = load_dataset("ItalyPowerDemand", merge_train_test=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
f = BaggingClassifier(RocketTreeClassifier(n_kernels=10))
f.fit(x_train, y_train)
print(f.score(x_test, y_test))

f = RocketClassifier()
f.fit(x_train, y_train)
print(f.score(x_test, y_test))
