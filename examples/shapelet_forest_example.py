import numpy as np
from sklearn.model_selection import cross_val_score

from wildboar import datasets
from wildboar.ensemble import ExtraShapeletTreesClassifier, ShapeletForestClassifier

x, y = datasets.load_gun_point()
extra = ExtraShapeletTreesClassifier(
    n_estimators=100, n_jobs=16, metric="scaled_euclidean"
)
rsf = ShapeletForestClassifier(n_estimators=100, n_jobs=16, metric="scaled_euclidean")
score = cross_val_score(rsf, x, y, cv=10)
print("RSF", np.mean(score))
score = cross_val_score(extra, x, y, cv=10)
print("Extra", np.mean(score))
