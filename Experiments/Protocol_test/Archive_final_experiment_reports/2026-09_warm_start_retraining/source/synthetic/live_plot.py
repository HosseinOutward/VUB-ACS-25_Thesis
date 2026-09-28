"""Refresh partial analysis while the detached warm-start runner is alive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from Experiments.warm_start_retraining.experiment import load_config


def alive(pid: int) -> bool:
    """Return whether a process still exists."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    """Render changed case outputs in disposable analyser processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--runner-pid", type=int, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    previous: tuple[tuple[str, int], ...] = ()
    while True:
        inputs = tuple(sorted(
            (str(path), path.stat().st_mtime_ns)
            for path in config.output_root.glob("cases/*/repeat_*/raw_stage_results.csv")
        ))
        if inputs != previous:
            result = subprocess.run(
                (sys.executable, "-m", "Experiments.warm_start_retraining.analyze", str(args.config.resolve())),
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                previous = inputs
            else:
                print(result.stdout + result.stderr, flush=True)
        if not alive(args.runner_pid):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
