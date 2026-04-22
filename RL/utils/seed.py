from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = False, env: Any | None = None) -> None:
    """
    Best-effort seeding across common libraries used in this repo.

    - python `random`
    - numpy
    - torch (if installed)
    - stable-baselines3 helper (if installed)
    - d3rlpy seed helper (if installed)
    - optional: a gymnasium env via `env.reset(seed=...)` when provided
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(seed)
    except Exception:
        pass

    try:
        import Offline_RL_algo.d3rlpy as d3rlpy  # local vendored copy

        d3rlpy.seed(seed)
    except Exception:
        pass

    if env is not None:
        try:
            env.reset(seed=seed)
        except Exception:
            pass

