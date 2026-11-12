"""Run the full research pipeline end to end.

    python run_pipeline.py                 # everything (primary + ablation)
    python run_pipeline.py --skip-download # reuse data/raw
    python run_pipeline.py --no-ablation

Each stage is a module that can also be run on its own (see README).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

STAGES = [
    ("download", "src.data.download"),
    ("inspect", "src.data.inspect"),
    ("preprocess", "src.data.preprocess"),
    ("sequences", "src.data.create_sequences"),
    ("train", "src.training.train"),
    ("evaluate", "src.training.evaluate"),
    ("explain", "src.explainability.feature_importance"),
]
ABLATION = "configs/ablation_level_target.yaml"


def run(module: str, env_extra: dict | None = None) -> None:
    env = {**os.environ, **(env_extra or {})}
    t0 = time.time()
    print(f"\n=== {module} {env_extra or ''} ===", flush=True)
    r = subprocess.run([sys.executable, "-m", module], env=env)
    if r.returncode != 0:
        sys.exit(f"Stage {module} failed (exit {r.returncode}); stopping - fix before continuing.")
    print(f"--- {module} OK in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()
    for name, mod in STAGES:
        if a.skip_download and name == "download":
            continue
        run(mod)
    if not a.no_ablation:
        for mod in ("src.data.create_sequences", "src.training.train", "src.training.evaluate"):
            run(mod, {"PI_CONFIG": ABLATION})
    run("src.reporting")
    run("src.export_dashboard")


if __name__ == "__main__":
    main()
