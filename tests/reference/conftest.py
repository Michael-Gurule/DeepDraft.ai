"""Shared helpers for reference tests.

Reference tests check an implementation against a closed form, an exact
small case, or an established library, so a wrong formula fails loudly
instead of producing a plausible-looking plot.
"""

import importlib.util
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(relpath: str):
    """Import a catalog module (for example "quantitative/garch.py") by path."""
    path = REPO_ROOT / relpath
    name = "ref_" + relpath.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def no_figures(monkeypatch, tmp_path):
    """Send any figures a function saves to a temporary directory."""
    monkeypatch.setenv("DEEPDRAFT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPDRAFT_NO_SHOW", "1")
