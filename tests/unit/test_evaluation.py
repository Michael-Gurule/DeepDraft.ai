import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from deepdraft.evaluation import (
    bootstrap_ci,
    mase,
    mean_confidence_interval,
    rolling_origin_splits,
    temporal_train_val_test_split,
    train_val_test_split,
)


def test_train_val_test_split_sizes_and_disjointness():
    X = np.arange(1000).reshape(-1, 1)
    y = np.arange(1000)
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split(X, y, val_size=0.2, test_size=0.1)
    assert (len(X_tr), len(X_va), len(X_te)) == (700, 200, 100)
    assert len(set(y_tr) | set(y_va) | set(y_te)) == 1000          # nothing lost or duplicated
    np.testing.assert_array_equal(X_tr[:, 0], y_tr)                # rows stay aligned


def test_train_val_test_split_stratifies_every_split():
    y = np.array([0] * 900 + [1] * 100)
    X = np.zeros((1000, 2))
    _, _, _, y_tr, y_va, y_te = train_val_test_split(X, y, stratify=y)
    for part in (y_tr, y_va, y_te):
        assert part.mean() == pytest.approx(0.1, abs=0.01)


def test_train_val_test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        train_val_test_split(np.zeros(10), val_size=0.5, test_size=0.5)


def test_temporal_split_is_ordered_and_complete():
    tr, va, te = temporal_train_val_test_split(100, val_size=0.2, test_size=0.1)
    assert tr.max() < va.min() and va.max() < te.min()
    assert len(tr) + len(va) + len(te) == 100


def test_rolling_origin_never_trains_on_the_future():
    folds = rolling_origin_splits(n_obs=50, initial=20, horizon=5)
    assert len(folds) == 6
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()
        assert len(test_idx) == 5
    assert [len(tr) for tr, _ in folds] == [20, 25, 30, 35, 40, 45]   # expanding window


def test_rolling_origin_max_splits_keeps_latest_folds():
    folds = rolling_origin_splits(n_obs=50, initial=20, horizon=5, max_splits=2)
    assert [te[0] for _, te in folds] == [40, 45]


def test_rolling_origin_rejects_short_series():
    with pytest.raises(ValueError):
        rolling_origin_splits(n_obs=10, initial=8, horizon=5)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=500)
    y_pred = y_true + rng.normal(scale=0.5, size=500)
    point, lo, hi = bootstrap_ci(mean_absolute_error, y_true, y_pred, n_boot=300)
    assert lo < point < hi
    assert point == pytest.approx(0.5 * np.sqrt(2 / np.pi), rel=0.1)   # E|N(0, 0.5^2)|


def test_bootstrap_ci_skips_undefined_resamples():
    from sklearn.metrics import roc_auc_score
    y_true = np.array([0] * 18 + [1] * 2)
    y_score = np.linspace(0, 1, 20)
    point, lo, hi = bootstrap_ci(roc_auc_score, y_true, y_score, n_boot=200)
    assert 0 <= lo <= point <= hi <= 1


def test_mase_is_one_for_the_naive_forecast():
    y_train = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    naive_errors_scale = np.mean(np.abs(np.diff(y_train)))
    y_true = np.array([6.0, 7.0])
    y_pred = y_true + naive_errors_scale                    # MAE equals the naive scale
    assert mase(y_true, y_pred, y_train) == pytest.approx(1.0)


def test_mase_seasonal_period():
    y_train = np.tile([1.0, 2.0, 3.0, 4.0], 3) + np.arange(12) * 0.1
    assert mase(y_train[-4:], y_train[-4:], y_train, m=4) == 0.0
    with pytest.raises(ValueError):
        mase([1.0], [1.0], [1.0, 2.0], m=4)


def test_mean_confidence_interval_matches_t_formula():
    from scipy import stats
    values = np.array([10.0, 12.0, 9.0, 11.0, 13.0])
    mean, lo, hi = mean_confidence_interval(values)
    half = stats.t.ppf(0.975, 4) * values.std(ddof=1) / np.sqrt(5)
    assert mean == pytest.approx(11.0)
    assert (lo, hi) == pytest.approx((11.0 - half, 11.0 + half))


def test_mean_confidence_interval_single_value_is_undefined():
    mean, lo, hi = mean_confidence_interval([3.0])
    assert mean == 3.0 and np.isnan(lo) and np.isnan(hi)
