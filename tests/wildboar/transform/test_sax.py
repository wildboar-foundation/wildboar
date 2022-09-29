# Authors: Isak Samsten
# License: BSD 3 clause

from wildboar.transform import PAA, SAX
from wildboar.utils.estimator_checks import check_estimator


def test_estimator_checks():
    check_estimator(SAX())
    check_estimator(PAA())
