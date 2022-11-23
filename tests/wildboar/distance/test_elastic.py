import pytest

from wildboar.datasets import load_two_lead_ecg
from wildboar.distance import pairwise_distance


@pytest.mark.parametrize(
    "metric", ["dtw", "erp", "lcss", "msm", "twe", "ddtw", "wdtw", "wddtw"]
)
def test_benchmark(benchmark, metric):
    X, y = load_two_lead_ecg()
    x = X[:100].reshape(-1).copy()
    y = X[100:200].reshape(-1).copy()

    benchmark(pairwise_distance, x, y, metric=metric, metric_params={"r": 1.0})
