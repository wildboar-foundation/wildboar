import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import check_random_state

from datasets import load_two_lead_ecg, load_dataset
from sklearn.model_selection import train_test_split

from datasets.outlier import kmeans_labeler
from wildboar.ensemble import IsolationShapeletForest

x, _y = load_dataset('Adiac', repository='ucr')
y = kmeans_labeler(x, random_state=0)
print(np.unique(y, return_counts=True))
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=10, test_size=0.2, stratify=y)
#
#
# # def make_sine(scale, min=-10, max=10, m=100, n=100, random=None):
# #     random = check_random_state(random)
# #     x = np.ones([n, m]) * np.linspace(min, max, m)
# #     x = np.sin(scale * x) + random.uniform(size=[n, m])
# #     return x
#
#
# # x_train = np.vstack([make_sine(1, n=100), make_sine(10, n=40)])
# # x_test = np.vstack([make_sine(10, n=5), make_sine(1, n=100)])
# # y_test = np.hstack([-np.ones(5), np.ones(100)])
#
f = IsolationShapeletForest(
    random_state=10,
    n_estimators=100,
    contamination='auto',
    metric="scaled_euclidean",
    n_jobs=-1,
    min_shapelet_size=0.2,
    max_shapelet_size=0.8)

f.fit(x_train, y_train)
print("Offset=%f" % f.offset_)
y_pred = f.predict(x_test)
print("BA offset=%f" % balanced_accuracy_score(y_test, y_pred))
