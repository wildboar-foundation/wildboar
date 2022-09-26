from wildboar.transform import (  # PivotEmbedding,; RandomShapeletEmbedding,
    IntervalTransform,
    RandomShapeletTransform,
    RocketTransform,
)
from wildboar.utils.estimator_checks import check_estimator


def test_estimator_checks():
    check_estimator(IntervalTransform())
    check_estimator(RandomShapeletTransform())
    check_estimator(RocketTransform())
    # TODO: fix failing tests
    # check_estimator(PivotEmbedding())
