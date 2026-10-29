"""Data splits that keep model selection away from the test set.

The rule every DeepDraft.ai module follows:

    train       fit the model
    validation  choose hyperparameters, thresholds, and early-stopping points
    test        report the final metric, once

Cross-validation inside the training split can replace the validation split.
For time series, every split respects time order: validation and test data
always come after the data the model learned from.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def train_val_test_split(*arrays, val_size: float = 0.2, test_size: float = 0.2,
                         stratify=None, random_state: int | None = 42):
    """Split arrays into train, validation, and test sets.

    Args:
        *arrays: arrays with the same first dimension (for example X, y).
        val_size: fraction of all samples for validation.
        test_size: fraction of all samples for testing.
        stratify: labels to preserve class proportions in every split.
        random_state: seed for the shuffle.

    Returns:
        For each input array, its train, validation, and test parts, in the
        same order scikit-learn uses: X_train, X_val, X_test, y_train, ...
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be in (0, 1) and sum to less than 1")

    # Carry the stratification labels through both splits as an extra array so
    # the second split can stratify on the remaining labels.
    arrays = list(arrays)
    if stratify is not None:
        arrays.append(np.asarray(stratify))

    first = train_test_split(*arrays, test_size=test_size, random_state=random_state,
                             stratify=stratify)
    rest, tests = first[0::2], first[1::2]

    val_fraction_of_rest = val_size / (1 - test_size)
    second = train_test_split(*rest, test_size=val_fraction_of_rest, random_state=random_state,
                              stratify=rest[-1] if stratify is not None else None)
    trains, vals = second[0::2], second[1::2]

    n_out = len(arrays) - (1 if stratify is not None else 0)
    out = []
    for tr, va, te in list(zip(trains, vals, tests))[:n_out]:
        out.extend([tr, va, te])
    return out


def temporal_train_val_test_split(n_obs: int, val_size: float = 0.15,
                                  test_size: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index arrays for a chronological train / validation / test split.

    The earliest observations train the model, the next block validates it,
    and the most recent block tests it. Nothing is shuffled.
    """
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1")
    n_test = max(1, int(round(n_obs * test_size)))
    n_val = max(1, int(round(n_obs * val_size)))
    n_train = n_obs - n_val - n_test
    if n_train < 1:
        raise ValueError("not enough observations for the requested split sizes")
    idx = np.arange(n_obs)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def rolling_origin_splits(n_obs: int, initial: int, horizon: int,
                          step: int | None = None,
                          max_splits: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window backtest folds for time series (rolling-origin evaluation).

    Fold k trains on observations [0, initial + k*step) and tests on the next
    `horizon` observations. Averaging the error over folds estimates
    out-of-sample accuracy far more reliably than a single holdout.

    Args:
        n_obs: length of the series.
        initial: size of the first training window.
        horizon: forecast horizon (test window length).
        step: how far the origin moves between folds (default: horizon).
        max_splits: keep only the most recent folds when set.

    Returns:
        A list of (train_idx, test_idx) pairs in chronological order.
    """
    step = horizon if step is None else step
    if initial < 1 or horizon < 1 or step < 1:
        raise ValueError("initial, horizon, and step must be positive")
    folds = []
    origin = initial
    while origin + horizon <= n_obs:
        folds.append((np.arange(origin), np.arange(origin, origin + horizon)))
        origin += step
    if not folds:
        raise ValueError("series too short for one fold: need initial + horizon <= n_obs")
    if max_splits is not None:
        folds = folds[-max_splits:]
    return folds
