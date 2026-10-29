"""Metrics with uncertainty, and forecasting metrics that compare to a baseline."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np


def bootstrap_ci(metric: Callable[[np.ndarray, np.ndarray], float],
                 y_true, y_pred, n_boot: int = 1000, alpha: float = 0.05,
                 random_state: int | None = 42) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for any metric(y_true, y_pred).

    Resamples test examples with replacement, recomputes the metric, and
    reports the alpha/2 and 1 - alpha/2 percentiles. Use it whenever a single
    test-set number would otherwise be reported without its uncertainty.

    Resamples where the metric is undefined (for example ROC-AUC on a sample
    containing one class) are skipped, whether the metric raises or returns NaN.

    Returns:
        (point_estimate, lower, upper)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)
    point = float(metric(y_true, y_pred))
    stats = []
    with warnings.catch_warnings():
        # Undefined resamples are expected and skipped below; their warnings
        # (e.g. scikit-learn's UndefinedMetricWarning) would only be noise.
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                value = float(metric(y_true[idx], y_pred[idx]))
            except ValueError:
                continue
            if np.isfinite(value):
                stats.append(value)
    if not stats:
        raise ValueError("metric was undefined on every bootstrap resample")
    lower, upper = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lower), float(upper)


def mase(y_true, y_pred, y_train, m: int = 1) -> float:
    """Mean absolute scaled error (Hyndman and Koehler, 2006).

    Divides the forecast MAE by the in-sample MAE of the seasonal naive
    forecast with period m (m=1 is the plain naive forecast). MASE < 1 means
    the model beats that naive baseline; it is scale-free and, unlike MAPE,
    well defined when the series is near zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        raise ValueError("y_train must be longer than the seasonal period m")
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale == 0:
        raise ValueError("naive in-sample errors are all zero; MASE is undefined")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)
