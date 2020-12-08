import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from wildboar.datasets import load_dataset
from wildboar.datasets.outlier import EmmottLabeler
from wildboar.ensemble import IsolationShapeletForest

x, y = load_dataset("SwedishLeaf", repository="wildboar/ucr", merge_train_test=True)
labeler = EmmottLabeler(
    n_outliers=0.05, difficulty=1, variation="tight", random_state=5
)
x, y = labeler.fit_transform(x, y)
print(np.unique(y, return_counts=True))
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=10, test_size=0.2, stratify=y
)

f = IsolationShapeletForest(
    random_state=10,
    n_estimators=100,
    contamination=0.1,
    metric="scaled_euclidean",
    n_jobs=-1,
    min_shapelet_size=0,
    max_shapelet_size=1,
)

f.fit(x_train, y_train)
y_pred = f.decision_function(x_test)
print("AUC=%f" % roc_auc_score(y_test, y_pred))


def plot_mds(x, y):
    import matplotlib.pylab as plt
    from sklearn.manifold import MDS

    m = MDS(n_components=2, random_state=1).fit_transform(x)
    m_normal = m[y == 1]
    m_outlier = m[y == -1]
    plt.scatter(m_normal[:, 0], m_normal[:, 1], c="blue")
    plt.scatter(m_outlier[:, 0], m_outlier[:, 1], c="red")
    plt.show()


plot_mds(x, y)
