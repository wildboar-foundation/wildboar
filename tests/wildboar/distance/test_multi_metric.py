import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from wildboar.distance._multi_metric import make_metric, make_metrics


def test_make_metric_missing_metric():
    with pytest.raises(ValueError):
        make_metric("missing_metric")


def test_make_metric_missing_parameter():
    with pytest.raises(TypeError):
        make_metric("dtw", max_missing_metric=1, min_missing_metric=0)


def test_make_metric_invalid_spec():
    with pytest.raises(ValueError, match="The parameter invalid_spec"):
        make_metric("dtw", invalid_spec=0)


def test_make_metric_missing_min_max():
    with pytest.raises(ValueError, match="The maximum"):
        make_metric("dtw", min_r=0.0)

    with pytest.raises(ValueError, match="The minimum"):
        make_metric("dtw", max_r=0.0)


def test_make_metric_single_spec():
    metrics = make_metric("dtw", min_r=0.0, max_r=0.25, default_n=10)
    assert len(metrics) == 10


def test_make_metric_muli_spec():
    metrics = make_metric(
        "wdtw", min_r=0.0, max_r=0.25, min_g=0.01, max_g=1, default_n=10
    )
    assert len(metrics) == 10 * 10


def test_make_metric_multi_spec_different_n():
    metrics = make_metric(
        "wdtw", num_r=3, min_r=0.0, max_r=0.25, min_g=0.01, max_g=1, default_n=10
    )
    assert len(metrics) == 3 * 10


def test_make_metrics():
    spec = [
        ("euclidean", None),
        ("dtw", None),
        ("dtw", dict(min_r=0.0, max_r=0.25, num_r=10)),
    ]

    metrics, weights = make_metrics(spec)

    assert len(metrics) == 1 + 1 + 10
    assert_almost_equal(
        weights,
        np.array(
            [
                0.333333333,
                0.333333333,
                0.0333333333,
                0.0333333333,
                0.0333333333,
                0.0333333333,
                0.0333333333,
                0.0333333333,
                0.0333333333,
                0.0333333333,
                0.0333333333,
                0.0333333333,
            ]
        ),
    )
