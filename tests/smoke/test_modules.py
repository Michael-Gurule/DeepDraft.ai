"""Smoke tests: every module's demo must run end to end in fast mode.

Each module runs as a script in a fresh subprocess with:
    DEEPDRAFT_FAST=1      shrink data and iterations (see deepdraft.core.runtime)
    MPLBACKEND=Agg        render figures without a display
    DEEPDRAFT_OUTPUT_DIR  write figures to a temporary directory

A module is skipped (not failed) when one of its top-level optional
dependencies is not installed, so `pip install -e ".[dev]"` runs the core
modules and `pip install -e ".[all,dev]"` runs everything.
"""

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = [
    "anomaly_detection", "bayesian", "causal_inference", "classic", "ensemble",
    "interpretability", "multi_agent_systems", "quantitative",
    "reinforcement_learning", "scientific_ml", "time_series",
]
OPTIONAL_DEPS = {
    "arch", "gymnasium", "hmmlearn", "lightgbm", "networkx", "prophet",
    "statsmodels", "torch", "xgboost",
}
TIMEOUT_S = int(os.environ.get("DEEPDRAFT_SMOKE_TIMEOUT", "300"))

MODULES = sorted(p for c in CATEGORIES for p in (ROOT / c).glob("*.py"))


def _required_optional_deps(path: Path) -> set[str]:
    """Optional packages imported at module top level (outside try blocks)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names & OPTIONAL_DEPS


@pytest.mark.parametrize("path", MODULES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_module_runs_in_fast_mode(path: Path, tmp_path: Path) -> None:
    missing = sorted(d for d in _required_optional_deps(path) if importlib.util.find_spec(d) is None)
    if missing:
        pytest.skip(f"optional dependencies not installed: {', '.join(missing)}")

    env = {
        **os.environ,
        "DEEPDRAFT_FAST": "1",
        "MPLBACKEND": "Agg",
        "DEEPDRAFT_NO_SHOW": "1",
        "DEEPDRAFT_OUTPUT_DIR": str(tmp_path / "outputs"),
        "PYTHONUNBUFFERED": "1",
    }
    # One thread per module: pytest-xdist already runs one module per core,
    # and letting PyTorch, BLAS, and joblib each spawn a thread per core
    # oversubscribes the machine by an order of magnitude.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        env.setdefault(var, "1")
    try:
        proc = subprocess.run(
            [sys.executable, str(path)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        tail = (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else ""
        pytest.fail(f"{path.name} exceeded {TIMEOUT_S}s in fast mode.\n{tail}")

    if proc.returncode != 0:
        pytest.fail(
            f"{path.name} exited with code {proc.returncode}.\n"
            f"--- stdout (tail) ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-4000:]}"
        )
