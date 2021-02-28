import numpy as np
import math
from wildboar.datasets import load_dataset
from wildboar.tree._tree import ShapeletTreeClassifier, RocketTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.model_selection import train_test_split


x, y = load_dataset("ItalyPowerDemand", merge_train_test=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
f = BaggingClassifier(RocketTreeClassifier(n_kernels=10))
f.fit(x_train, y_train)
print(f.score(x_test, y_test))
