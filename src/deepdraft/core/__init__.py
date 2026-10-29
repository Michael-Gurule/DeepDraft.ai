"""Shared runtime helpers used by every DeepDraft.ai module."""

from deepdraft.core.plotting import save_figure
from deepdraft.core.runtime import fast_or, is_fast

__all__ = ["fast_or", "is_fast", "save_figure"]
