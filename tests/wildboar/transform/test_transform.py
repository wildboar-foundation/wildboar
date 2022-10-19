# Authors: Isak Samsten
# License: BSD 3 clause

from wildboar.transform import (  # PivotEmbedding,; RandomShapeletEmbedding,
    IntervalTransform,
    MatrixProfileTransform,
    RandomShapeletTransform,
    RocketTransform,
)
from wildboar.utils.estimator_checks import check_estimator


def test_estimator_checks():
    check_estimator(IntervalTransform())
    check_estimator(RandomShapeletTransform())
    check_estimator(RocketTransform())
    check_estimator(MatrixProfileTransform())
    # TODO: fix failing tests
    # check_estimator(PivotEmbedding())
