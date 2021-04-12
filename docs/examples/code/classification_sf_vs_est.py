import numpy as np
from sklearn.model_selection import cross_validate

from wildboar.datasets import load_dataset
from wildboar.ensemble import ExtraShapeletTreesClassifier, ShapeletForestClassifier

random_state = 1234

x, y = load_dataset("Beef")

classifiers = {
    "Shapelet forest": ShapeletForestClassifier(
        n_shapelets=10,
        metric="scaled_euclidean",
        n_jobs=-1,
        random_state=random_state,
    ),
    "Extra Shapelet Trees": ExtraShapeletTreesClassifier(
        metric="scaled_euclidean",
        n_jobs=-1,
        random_state=random_state,
    ),
}

for name, clf in classifiers.items():
    score = cross_validate(clf, x, y, scoring="roc_auc_ovo", n_jobs=1)
    print("Classifier: %s" % name)
    print(" - fit-time:   %.2f" % np.mean(score["fit_time"]))
    print(" - test-score: %.2f" % np.mean(score["test_score"]))
