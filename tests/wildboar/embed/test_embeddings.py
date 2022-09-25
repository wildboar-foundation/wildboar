from wildboar.embed import (  # PivotEmbedding,; RandomShapeletEmbedding,
    IntervalEmbedding,
    RocketEmbedding,
)
from wildboar.utils.estimator_checks import check_estimator


def test_estimator_checks():
    check_estimator(IntervalEmbedding())
    # TODO: fix failing tests
    # check_estimator(RandomShapeletEmbedding())
    check_estimator(RocketEmbedding())
    # check_estimator(PivotEmbedding())
