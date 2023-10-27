import itertools
import re
from collections import defaultdict

import numpy as np

from ..utils.validation import check_option
from ._distance import _METRICS, _SUBSEQUENCE_METRICS


def parse_metric_spec(kwargs):
    specs = defaultdict(dict)
    for key, value in kwargs.items():
        m = re.match(r"^(num|max|min)_([a-zA-Z_$]\w*)$", key)
        if m:
            specs[m.group(2)][m.group(1)] = value
        else:
            raise ValueError(
                f"The parameter {key} must be prefixed with "
                f"'min_', 'max_' or 'num_', got {key} "
            )

    return specs


def make_parameter_grid(metric_spec, default_n=10):
    if metric_spec is None:
        return [{}]

    specs = parse_metric_spec(metric_spec)
    params = []
    grids = []
    for param, spec in specs.items():
        if "max" not in spec:
            raise ValueError(f"The maximum value is missing for {param}.")

        if "min" not in spec:
            raise ValueError(f"The minimum value is missing for {param}.")

        if "num" in spec:
            num = spec["num"]
        else:
            num = default_n

        params.append(param)
        grids.append(np.linspace(spec["min"], spec["max"], num))

    return [
        {param: value for param, value in zip(params, grid)}
        for grid in itertools.product(*grids)
    ]


def _make_metric(metrics, metric, default_n=10, **kwargs):
    Metric = check_option(metrics, metric, "metric")
    if kwargs is None or default_n is None:
        return [Metric()]
    else:
        parameter_grid = make_parameter_grid(kwargs, default_n=default_n)
        return [Metric(**metric_params) for metric_params in parameter_grid]


def make_subsequence_metric(metric, *, default_n=10, **kwargs):
    return _make_metric(_SUBSEQUENCE_METRICS, metric, default_n=default_n, **kwargs)


def make_metric(metric, *, default_n=10, **kwargs):
    return _make_metric(_METRICS, metric, default_n=default_n, **kwargs)


def _make_metrics(metric_specs, factory):
    metrics = []
    weights = []
    base_weight = 1.0 / len(metric_specs)

    if isinstance(metric_specs, dict):
        metric_specs = metric_specs.items()
    for metric_name, metric_spec in metric_specs:
        metric_spec = {} if metric_spec is None else metric_spec
        current_metrics = factory(metric_name, **metric_spec)
        for metric in current_metrics:
            metrics.append(metric)
            weights.append(base_weight / len(current_metrics))

    return metrics, np.array(weights, dtype=float)


def make_metrics(metric_specs):
    return _make_metrics(metric_specs, make_metric)


def make_subsequence_metrics(metric_specs):
    return _make_metrics(metric_specs, make_subsequence_metric)
