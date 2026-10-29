"""Runtime switches controlled by environment variables.

DEEPDRAFT_FAST=1
    Shrink datasets, iteration counts, and search grids so a module's demo
    finishes in seconds. The smoke tests run every module this way. Results
    in fast mode are only a check that the code runs, not meaningful metrics.
"""

import os
from typing import TypeVar

T = TypeVar("T")

_TRUTHY = {"1", "true", "yes", "on"}


def is_fast() -> bool:
    """Return True when DEEPDRAFT_FAST is set to a truthy value."""
    return os.environ.get("DEEPDRAFT_FAST", "").strip().lower() in _TRUTHY


def fast_or(normal: T, fast: T) -> T:
    """Return `fast` in fast mode and `normal` otherwise.

    Example:
        n_samples = fast_or(5000, 300)
    """
    return fast if is_fast() else normal
