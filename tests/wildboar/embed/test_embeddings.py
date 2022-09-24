from wildboar.embed import (
    IntervalEmbedding,
    PivotEmbedding,
    RandomShapeletEmbedding,
    RocketEmbedding,
)
from wildboar.utils.estimator_checks import check_estimator


def test_estimator_checks():
    check_estimator(IntervalEmbedding())
    check_estimator(RandomShapeletEmbedding())
    check_estimator(RocketEmbedding())
    check_estimator(PivotEmbedding())
