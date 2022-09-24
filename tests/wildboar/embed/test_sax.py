from wildboar.embed import PAA, SAX
from wildboar.utils.estimator_checks import check_estimator


def test_estimator_checks():
    check_estimator(SAX())
    check_estimator(PAA())
