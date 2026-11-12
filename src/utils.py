"""Shared helpers: configuration loading, logging, seeding and paths."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load the experiment configuration.

    Resolution order: explicit `path`, env var PI_CONFIG, project config.yaml.
    A config file may contain `base: <file>`; it is then deep-merged over that
    base (used for ablation experiments that only override a few keys).
    """
    path = Path(path or os.environ.get("PI_CONFIG") or PROJECT_ROOT / "config.yaml")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        cfg = _deep_merge(load_config(PROJECT_ROOT / cfg.pop("base")), cfg)
    return cfg


def exp_paths(cfg: dict) -> dict[str, Path]:
    """Experiment-specific directories (windows, models, results), created if absent."""
    p = {
        "windows": project_path(cfg["sequences"].get("windows_dir", cfg["data"]["processed_dir"])),
        "models": project_path(cfg["training"]["model_dir"]),
        "results": project_path(cfg["training"]["results_dir"]),
    }
    p["graphs"], p["metrics"], p["reports"] = (p["results"] / x for x in ("graphs", "metrics", "reports"))
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Return a logger writing to stdout and (optionally) logs/<log_file>."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        log_dir = project_path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility (CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
