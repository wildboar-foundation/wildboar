from numpy.testing import assert_almost_equal
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from wildboar.datasets import load_two_lead_ecg, outlier
from wildboar.ensemble import IsolationShapeletForest
from wildboar.utils.estimator_checks import check_estimator


def test_estimator_checks():
    check_estimator(IsolationShapeletForest())


def load_test_dataset():
    X, y = load_two_lead_ecg()
    X, y = outlier.emmott_outliers(X, y, n_outliers=0.05, random_state=0)
    return train_test_split(X, y, random_state=0)


def test_fit_predict_score_sample():
    X_train, X_test, y_train, y_test = load_test_dataset()
    isf = IsolationShapeletForest(n_shapelets=1, random_state=0)
    isf.fit(X_train)

    expected_score_samples = [
        -0.5346257,
        -0.4549916,
        -0.4623454,
        -0.4922771,
        -0.4988597,
        -0.4394299,
        -0.4887521,
        -0.4621849,
        -0.4910222,
        -0.4955371,
    ]
    assert_almost_equal(isf.score_samples(X_test[:10]), expected_score_samples)
    assert_almost_equal(
        balanced_accuracy_score(y_test, isf.predict(X_test)), 0.9071428571428571
    )
