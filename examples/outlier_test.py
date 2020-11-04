import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import check_random_state

from datasets import load_two_lead_ecg
from model_selection.outlier import train_test_split
from wildboar.ensemble import IsolationShapeletForest

# def load_data(train_file, test_file):
#     train = np.loadtxt(train_file, delimiter=",")
#     test = np.loadtxt(test_file, delimiter=",")
#     train = np.vstack([train, test])
#     y = train[:, 0].astype(np.float64)
#     y = np.ascontiguousarray(y)
#     x = np.ascontiguousarray(train[:, 1:].astype(np.float64))
#     return x, y


# def split_train_test(x, y, normal_class, random_state=None, test_size=0.2, anomalies_test_size=0.05):
#     random_state = check_random_state(random_state)
#     normal = y == normal_class
#     y = y.copy()
#     y[normal] = 1
#     y[~normal] = -1
#
#     x_normal = x[np.where(y == 1)]
#     x_anomalous = x[np.where(y == -1)]
#     y_normal = y[np.where(y == 1)]
#     y_anomalous = y[np.where(y == -1)]
#
#     x_normal_train, x_normal_test, y_normal_train, y_normal_test = train_test_split(
#         x_normal, y_normal, test_size=test_size, random_state=random_state
#     )
#
#     n_sample = min(x_anomalous.shape[0], x_normal.shape[0])
#     idx = sample_without_replacement(x_anomalous.shape[0], n_sample, random_state=random_state)
#     n_training_anomalies = math.ceil(x_normal_train.shape[0] * anomalies_test_size)
#     idx = idx[:n_training_anomalies]
#
#     x_anomalous_train = x_anomalous[idx, :]
#     y_anomalous_train = y_anomalous[idx]
#
#     x_anomalous_test = np.delete(x_anomalous, idx, axis=0)
#     y_anomalous_test = np.delete(y_anomalous, idx)
#
#     x_train = np.vstack([x_normal_train, x_anomalous_train])
#     y_train = np.hstack([y_normal_train, y_anomalous_train])
#     x_test = np.vstack([x_normal_test, x_anomalous_test])
#     y_test = np.hstack([y_normal_test, y_anomalous_test])
#     return x_train, x_test, y_train, y_test


x, y = load_two_lead_ecg()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 1, random_state=10, test_size=0.2, anomalies_train_size=0.05)


def make_sine(scale, min=-10, max=10, m=100, n=100, random=None):
    random = check_random_state(random)
    x = np.ones([n, m]) * np.linspace(min, max, m)
    x = np.sin(scale * x) + random.uniform(size=[n, m])
    return x


# x_train = np.vstack([make_sine(1, n=100), make_sine(10, n=40)])
# x_test = np.vstack([make_sine(10, n=5), make_sine(1, n=100)])
# y_test = np.hstack([-np.ones(5), np.ones(100)])

f = IsolationShapeletForest(
    random_state=10,
    n_estimators=10000,
    contamination="auc",
    contamination_set="training",
    metric="scaled_euclidean",
    bootstrap=False,
    n_jobs=-1,
    min_shapelet_size=0.2,
    max_shapelet_size=0.8)

# f = IsolationForest(n_estimators=1000000, n_jobs=-1)

f.fit(x_train, y_train)
print("Offset=%f" % f.offset_)
print("BA offset=%f" % balanced_accuracy_score(y_test, f.predict(x_test)))
#
# x_train, y_train = x_test, y_test
# y_prob = f._oob_score_samples(x_train)
# y_prob = f.score_samples(x_train)
# ba_score = ba(y_train, y_prob)
#
# ba_score_offset = y_prob[np.argmax(ba_score)]
# print("ba_score_offset=%f" % ba_score_offset)
# f.offset_ = ba_score_offset
# print("BA ba_score_offset=%f" % balanced_accuracy_score(y_test, f.predict(x_test)))

# fpr, tpr, auc_thresholds = roc_curve(y_train, y_prob)
# auc_offset = auc_thresholds[np.argmax(tpr - fpr)]
# print('AUC Offset=%f' % auc_offset)
#
# precision, recall, prc_thresholds = precision_recall_curve(y_train, y_prob)
# fscore = (2 * precision * recall) / (precision + recall)
# prc_offset = prc_thresholds[np.argmax(fscore)]
# f.offset_ = prc_offset
# print('PRC Offset=%f' % prc_offset)
# print("BA prc_offset=%f" % balanced_accuracy_score(y_test, f.predict(x_test)))
