from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from datasets.outlier import EmmottLabeler
from wildboar.ensemble import IsolationShapeletForest

x, y = load_dataset('SwedishLeaf', repository='wildboar/ucr', merge_train_test=True)
labeler = EmmottLabeler(n_outliers=0.05, difficulty=1, variation='tight', random_state=5)
x, y = labeler.fit_transform(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.2, stratify=y)

f = IsolationShapeletForest(
    random_state=10,
    n_estimators=100,
    contamination=0.1,
    metric="scaled_euclidean",
    n_jobs=-1,
    min_shapelet_size=0,
    max_shapelet_size=1)

f.fit(x_train, y_train)
y_pred = f.decision_function(x_test)
print("AUC=%f" % roc_auc_score(y_test, y_pred))
