"""Continuously refresh sparse-replay plots: one experiment's, or a directory's global view.

`watch` follows one experiment and exits with it; `watch_global` follows a directory of experiments
and runs until it is stopped, because experiments there start and finish at different times. Each
refresh runs the plotting module in a new interpreter, so edits to plotting code are picked up
without restarting the watcher. Failed refreshes leave the last successful analysis intact.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import sleep
from typing import Any

from Experiments.Protocol_test import plot


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLOT_SOURCES = (
    Path(__file__).with_name("plot.py"),
    Path(__file__).with_name("records.py"),
    Path(__file__).with_name("rd_metrics.py"),
)
_RESULT_PATTERNS = (
    "test_config.json",
    "cases/*/*/compression_records.csv",
    "cases/*/*/complete.json",
)
# The same inputs recursively, wherever the experiment roots happen to sit.
_GLOBAL_RESULT_PATTERNS = tuple(f"**/{pattern}" for pattern in _RESULT_PATTERNS)
REFRESH_SECONDS = 60
RENDER_TIMEOUT_SECONDS = 10 * 60


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def source_digest(paths: tuple[Path, ...] = _PLOT_SOURCES) -> str:
    """Hash plotting sources so a saved edit schedules a fresh interpreter run."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def result_digest(output_root: Path, patterns: tuple[str, ...] = _RESULT_PATTERNS) -> str:
    """Hash the names and stable metadata of every input that can alter the plots."""
    paths = sorted({path for pattern in patterns for path in output_root.glob(pattern)})
    digest = hashlib.sha256()
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        digest.update(str(path.relative_to(output_root)).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def process_start_ticks(pid: int) -> int | None:
    """Return Linux process start ticks, or None when the process no longer exists."""
    try:
        fields_after_name = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return int(fields_after_name[19])


def _runner_alive(pid: int, expected_start_ticks: int) -> bool:
    return process_start_ticks(pid) == expected_start_ticks


def _publish(staging: Path, output: Path) -> None:
    """Replace analysis only after a complete render, restoring it if publication fails."""
    previous = output.with_name(f".{output.name}.previous")
    if previous.exists():
        if output.exists():
            shutil.rmtree(previous)
        else:
            previous.replace(output)
    if output.exists():
        output.replace(previous)
    try:
        staging.replace(output)
    except OSError:
        if previous.exists() and not output.exists():
            previous.replace(output)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def _refresh_once(target: Path, output: Path, control_dir: Path, *, global_view: bool) -> bool:
    """Render `target` in a disposable directory and atomically publish only on success."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-refresh-", dir=output.parent))
    command = (
        sys.executable, "-u", "-m", "Experiments.Protocol_test.plot", str(target),
        *(("--global",) if global_view else ()), "--out", str(staging),
    )
    try:
        result = subprocess.run(
            command, cwd=_REPO_ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=RENDER_TIMEOUT_SECONDS,
        )
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
        if result.returncode:
            _atomic_json(control_dir / "plot_refresh.json", {
                "status": "retrying", "timestamp": datetime.now().astimezone().isoformat(),
                "returncode": result.returncode, "command": list(command),
                "message": "Plot refresh failed; the last successful analysis was preserved.",
            })
            return False
        _publish(staging, output)
        _atomic_json(control_dir / "plot_refresh.json", {
            "status": "updated", "timestamp": datetime.now().astimezone().isoformat(),
            "returncode": 0, "command": list(command),
        })
        return True
    except (OSError, subprocess.TimeoutExpired) as error:
        _atomic_json(control_dir / "plot_refresh.json", {
            "status": "retrying", "timestamp": datetime.now().astimezone().isoformat(),
            "message": f"Plot refresh could not finish: {error}",
        })
        return False
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def watch(output_root: Path, runner_pid: int, control_dir: Path, interval: int = REFRESH_SECONDS) -> int:
    """Refresh changed results until the original experiment process exits, including final state."""
    output_root, control_dir = output_root.resolve(), control_dir.resolve()
    control_dir.mkdir(parents=True, exist_ok=True)
    start_ticks = process_start_ticks(runner_pid)
    if start_ticks is None:
        _atomic_json(control_dir / "plot_refresh.json", {
            "status": "stopped", "stopped": datetime.now().astimezone().isoformat(),
            "reason": f"experiment_process_{runner_pid}_already_exited",
        })
        return 0
    while not output_root.is_dir():
        _atomic_json(control_dir / "plot_refresh.json", {
            "status": "waiting_for_output", "timestamp": datetime.now().astimezone().isoformat(),
        })
        if not _runner_alive(runner_pid, start_ticks):
            _atomic_json(control_dir / "plot_refresh.json", {
                "status": "stopped", "stopped": datetime.now().astimezone().isoformat(),
                "reason": "experiment_process_exited_before_creating_output",
            })
            return 0
        sleep(min(interval, 5))
    last_successful_signature: tuple[str, str] | None = None
    while True:
        alive = _runner_alive(runner_pid, start_ticks)
        try:
            signature = (result_digest(output_root), source_digest())
        except OSError as error:
            _atomic_json(control_dir / "plot_refresh.json", {
                "status": "retrying", "timestamp": datetime.now().astimezone().isoformat(),
                "message": f"Plot inputs changed while being read: {error}",
            })
            if not alive:
                break
            sleep(interval)
            continue
        has_records = any(output_root.glob("cases/*/*/compression_records.csv"))
        if has_records and signature != last_successful_signature:
            if _refresh_once(output_root, output_root / "analysis", control_dir, global_view=False):
                last_successful_signature = signature
        elif not has_records:
            _atomic_json(control_dir / "plot_refresh.json", {
                "status": "waiting_for_records",
                "timestamp": datetime.now().astimezone().isoformat(),
            })
        if not alive:
            break
        sleep(interval)
    state_path = control_dir / "plot_refresh.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    _atomic_json(state_path, state | {
        "status": "stopped", "stopped": datetime.now().astimezone().isoformat(),
        "reason": "experiment_process_exited",
    })
    return 0


def watch_global(container: Path, control_dir: Path, interval: int = REFRESH_SECONDS) -> int:
    """Refresh the global view of every experiment under `container` until this watcher is stopped.

    Unlike `watch`, no single process owns this view: experiments appear, finish and are added under
    `container` over time, so it has no completion to exit on.
    """
    container, control_dir = container.resolve(), control_dir.resolve()
    control_dir.mkdir(parents=True, exist_ok=True)
    output = container / plot.GLOBAL_ANALYSIS_DIR
    last_successful_signature: tuple[str, str] | None = None
    while True:
        try:
            signature = (result_digest(container, _GLOBAL_RESULT_PATTERNS), source_digest())
        except OSError as error:
            _atomic_json(control_dir / "plot_refresh.json", {
                "status": "retrying", "timestamp": datetime.now().astimezone().isoformat(),
                "message": f"Global plot inputs changed while being read: {error}",
            })
            sleep(interval)
            continue
        if any(container.glob("**/cases/*/*/compression_records.csv")):
            if signature != last_successful_signature and _refresh_once(
                    container, output, control_dir, global_view=True):
                last_successful_signature = signature
        else:
            _atomic_json(control_dir / "plot_refresh.json", {
                "status": "waiting_for_records", "timestamp": datetime.now().astimezone().isoformat(),
            })
        sleep(interval)


def main() -> None:
    """Run the per-experiment watcher used by the supervisor, or the standalone global watcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path,
                        help="One experiment output root, or the directory holding several with --global.")
    parser.add_argument("--global", dest="global_view", action="store_true",
                        help="Watch every experiment under `target` and refresh their shared global view.")
    parser.add_argument("--runner-pid", type=int, help="Experiment process this watcher exits with.")
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--interval", type=int, default=REFRESH_SECONDS)
    args = parser.parse_args()
    assert args.interval > 0, "Plot refresh interval must be positive."
    if args.global_view:
        assert args.runner_pid is None, "The global watcher outlives any single experiment process."
        control_dir = args.control_dir or args.target / f"{plot.GLOBAL_ANALYSIS_DIR}_control"
        raise SystemExit(watch_global(args.target, control_dir, args.interval))
    assert args.runner_pid is not None and args.control_dir is not None, (
        "Watching one experiment requires --runner-pid and --control-dir.")
    raise SystemExit(watch(args.target, args.runner_pid, args.control_dir, args.interval))


if __name__ == "__main__":
    main()
