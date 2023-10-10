from wildboar.distance import KMeans, KNeighbourClassifier
from wildboar.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(KNeighbourClassifier())
    check_estimator(KMeans())
