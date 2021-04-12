import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier

from wildboar.datasets import list_datasets, load_dataset
from wildboar.ensemble import ExtraShapeletTreesClassifier, ShapeletForestClassifier

random_state = 1234
classifiers = {
    "Nearest neighbors": KNeighborsClassifier(
        n_neighbors=1,
        metric="euclidean",
    ),
    "Shapelet forest": ShapeletForestClassifier(
        n_shapelets=10,
        metric="scaled_euclidean",
        random_state=random_state,
        n_jobs=-1,
    ),
    "Extra shapelet trees": ExtraShapeletTreesClassifier(
        metric="scaled_euclidean",
        n_jobs=-1,
        random_state=random_state,
    ),
}

repository = "wildboar/ucr-tiny"
datasets = list_datasets(repository)
df = pd.DataFrame(columns=classifiers.keys(), index=datasets, dtype=np.float)
for dataset in datasets:
    print(dataset)
    x, y = load_dataset(dataset, repository=repository)
    for clf_name, clf in classifiers.items():
        print(" ", clf_name)
        score = cross_validate(clf, x, y, scoring="roc_auc_ovo", n_jobs=1)
        df.loc[dataset, clf_name] = np.mean(score["test_score"])

df.to_csv("../tab/classification_cmp.csv", float_format="%.3f")
