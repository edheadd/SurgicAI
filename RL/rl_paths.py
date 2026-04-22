"""
Centralized path + config helpers for the `RL/` package.

This module provides:
- a canonical `RL` directory (derived from this file location)
- environment-variable overrides for repo/data roots
- helpers to build standard experiment directories
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_path(var_name: str) -> Optional[Path]:
    value = os.environ.get(var_name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def rl_dir() -> Path:
    """Absolute path to the `RL/` directory."""
    return Path(__file__).resolve().parent


def repo_dir() -> Path:
    """
    Best-effort absolute path to the repository root.

    By default, this assumes `RL/` lives directly under the repo root.
    Override with `SURGICAI_ROOT` if your layout differs.
    """
    override = _env_path("SURGICAI_ROOT")
    if override is not None:
        return override
    return rl_dir().parent


def data_dir() -> Path:
    """
    Default location for data artifacts (expert trajectories, images, etc.).

    Override with `SURGICAI_DATA_DIR` to redirect all run outputs.
    """
    override = _env_path("SURGICAI_DATA_DIR")
    if override is not None:
        return override
    # Keep artifacts colocated with code by default for backwards compatibility.
    return rl_dir()


@dataclass(frozen=True)
class ExperimentKey:
    task_name: str
    algorithm: str
    reward_type: str
    seed: int
    variant: str = "base_env"


def experiment_dir(key: ExperimentKey) -> Path:
    """
    Standard experiment directory layout.

    Layout:
      <data_dir>/<task>/<algorithm>/<reward_type>/seed_<seed>/<variant>/
    """
    return (
        data_dir()
        / str(key.task_name)
        / str(key.algorithm)
        / str(key.reward_type)
        / f"seed_{int(key.seed)}"
        / str(key.variant)
    )


def checkpoints_dir(key: ExperimentKey) -> Path:
    """Standard directory for SB3 checkpoints."""
    return experiment_dir(key) / "checkpoints"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

