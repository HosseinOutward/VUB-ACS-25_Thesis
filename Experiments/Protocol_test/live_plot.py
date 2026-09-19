"""Continuously refresh sparse-replay plots while its experiment process is alive.

Each refresh runs the plotting module in a new interpreter, so edits to plotting code are picked
up without restarting this watcher. Failed refreshes leave the last successful analysis intact.
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


def result_digest(output_root: Path) -> str:
    """Hash the names and stable metadata of every input that can alter the plots."""
    paths = sorted({path for pattern in _RESULT_PATTERNS for path in output_root.glob(pattern)})
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


def refresh_once(output_root: Path, control_dir: Path) -> bool:
    """Render in a disposable directory and atomically publish only a successful result."""
    staging = Path(tempfile.mkdtemp(prefix=".analysis-refresh-", dir=output_root))
    command = (
        sys.executable, "-u", "-m", "Experiments.Protocol_test.plot", str(output_root),
        "--out", str(staging),
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
        _publish(staging, output_root / "analysis")
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
            if refresh_once(output_root, control_dir):
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


def main() -> None:
    """Run the live plot watcher used by the detached experiment supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=REFRESH_SECONDS)
    args = parser.parse_args()
    assert args.interval > 0, "Plot refresh interval must be positive."
    raise SystemExit(watch(args.output_root, args.runner_pid, args.control_dir, args.interval))


if __name__ == "__main__":
    main()
