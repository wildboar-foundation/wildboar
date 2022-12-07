# Authors: Isak Samsten
# License: BSD 3 clause
import pytest

from wildboar.transform import (
    PAA,
    SAX,
    IntervalTransform,
    MatrixProfileTransform,
    PivotTransform,
    RandomShapeletTransform,
    RocketTransform,
)
from wildboar.utils._testing import (
    assert_exhaustive_parameter_checks,
    assert_parameter_checks,
)
from wildboar.utils.estimator_checks import check_estimator


@pytest.mark.parametrize(
    "estimator",
    [
        IntervalTransform(),
        RandomShapeletTransform(),
        RocketTransform(),
        MatrixProfileTransform(),
        PivotTransform(),
        SAX(),
        PAA(),
    ],
)
def test_estimator_checks(estimator):
    check_estimator(estimator)
    assert_exhaustive_parameter_checks(estimator)
    assert_parameter_checks(estimator)
