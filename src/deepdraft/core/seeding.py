"""One call to make a demo reproducible."""

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> np.random.Generator:
    """Seed Python, NumPy, and (when installed) PyTorch; return a NumPy Generator.

    Prefer the returned Generator (``rng.normal(...)``) for new code. The legacy
    global NumPy state is also seeded so older ``np.random.*`` calls and
    libraries that use it are reproducible too.

    PYTHONHASHSEED is set for any subprocesses this process starts; it cannot
    change hash randomization for the already-running interpreter.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
