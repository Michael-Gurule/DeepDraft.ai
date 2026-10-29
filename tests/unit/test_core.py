import random

import matplotlib
import numpy as np
import pytest

from deepdraft.core import fast_or, is_fast, save_figure, seed_everything

matplotlib.use("Agg")


def test_seed_everything_makes_numpy_and_random_repeatable():
    rng_a = seed_everything(7)
    a = (random.random(), np.random.rand(), rng_a.normal())
    rng_b = seed_everything(7)
    b = (random.random(), np.random.rand(), rng_b.normal())
    assert a == b


def test_seed_everything_seeds_torch_when_installed():
    torch = pytest.importorskip("torch")
    seed_everything(3)
    a = torch.rand(3)
    seed_everything(3)
    assert torch.equal(a, torch.rand(3))


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("0", False), ("", False)])
def test_fast_mode_switch(monkeypatch, value, expected):
    monkeypatch.setenv("DEEPDRAFT_FAST", value)
    assert is_fast() is expected
    assert fast_or("normal", "fast") == ("fast" if expected else "normal")


def test_save_figure_writes_to_output_dir_and_closes(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setenv("DEEPDRAFT_OUTPUT_DIR", str(tmp_path / "figs"))
    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    path = save_figure("line.png")
    assert path == tmp_path / "figs" / "line.png"
    assert path.exists() and path.stat().st_size > 0
    assert not plt.fignum_exists(fig.number)
