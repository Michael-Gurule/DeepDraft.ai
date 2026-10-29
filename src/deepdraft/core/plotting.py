"""Figure output that works the same in a terminal, a notebook, and CI.

DEEPDRAFT_OUTPUT_DIR
    Directory for saved figures (default: ./outputs). Created on first use.
DEEPDRAFT_NO_SHOW=1
    Save figures without opening a window. Figures are also never shown when
    matplotlib uses a non-interactive backend such as Agg.
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

_NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def output_dir() -> Path:
    """Directory where figures are written."""
    path = Path(os.environ.get("DEEPDRAFT_OUTPUT_DIR", "outputs"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _should_show() -> bool:
    if os.environ.get("DEEPDRAFT_NO_SHOW", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS


def save_figure(filename: str, fig=None, dpi: int = 150, **savefig_kwargs) -> Path:
    """Save a figure to the output directory, then show or close it.

    Replaces the pattern `plt.savefig(name); plt.show()`. Showing is skipped
    in headless runs so scripts never block, and figures are closed afterward
    so long demos do not accumulate open figures.

    Args:
        filename: file name such as "pca_loadings.png" (placed in output_dir()).
        fig: figure to save; defaults to the current figure.
        dpi: resolution passed to savefig.
        **savefig_kwargs: any other matplotlib savefig arguments.

    Returns:
        Path to the saved file.
    """
    fig = fig if fig is not None else plt.gcf()
    path = output_dir() / Path(filename).name
    fig.savefig(path, dpi=dpi, **savefig_kwargs)
    if _should_show():
        plt.show()
    plt.close(fig)
    return path
