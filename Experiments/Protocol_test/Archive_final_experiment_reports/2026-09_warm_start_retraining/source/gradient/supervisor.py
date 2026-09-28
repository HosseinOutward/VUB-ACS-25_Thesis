"""Detached supervisor for the reduced-epoch gradient warm-start experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic, sleep
from typing import Any

from Experiments.warm_start_gradient_retraining.experiment import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECK_DELAYS = (60, 4 * 60, 25 * 60)
ROUTINE_DELAY = 2 * 60 * 60


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one supervisor state file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def snapshot(config_path: Path, runner: subprocess.Popen[bytes], elapsed: int,
             runner_log: Path) -> dict[str, Any]:
    """Capture process, case, error, memory, and GPU health."""
    config = load_config(config_path)
    text = runner_log.read_text(errors="replace") if runner_log.exists() else ""
    memory = {
        fields[0].rstrip(":"): int(fields[1]) // 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if (fields := line.split())[0].rstrip(":") in {"MemAvailable", "SwapFree"}
    }
    gpu = subprocess.run(
        ("nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu",
         "--format=csv,noheader"), capture_output=True, text=True).stdout.splitlines()
    return {
        "timestamp": datetime.now().astimezone().isoformat(), "elapsed_seconds": elapsed,
        "runner_pid": runner.pid, "runner_alive": runner.poll() is None,
        "returncode": runner.poll(),
        "completed_repeats": len(tuple(config.output_root.glob("cases/repeat_*/complete.json"))),
        "expected_repeats": config.training_repeats,
        "trace_ready": (config.output_root / "trace" / "required_gradients.pt").exists(),
        "parent_ready": (config.output_root / "parent" / "round12_parent.pt").exists(),
        "error_markers": [marker for marker in ("Traceback", "out of memory", "CUDA error")
                          if marker.lower() in text.lower()],
        **memory, "gpu_status": gpu,
    }


def supervise(config_path: Path) -> int:
    """Run the study independently and persist repository-standard health checks."""
    config = load_config(config_path)
    control = config.output_root.parent / f"{config.output_root.name}_control"
    control.mkdir(parents=True, exist_ok=True)
    runner_log, health = control / "runner.log", control / "health_checks.jsonl"
    state = control / "supervisor.json"
    started = datetime.now().astimezone().isoformat()
    with runner_log.open("ab", buffering=0) as output:
        runner = subprocess.Popen(
            (sys.executable, "-u", "-m", "Experiments.warm_start_gradient_retraining.experiment",
             str(config_path)), cwd=PROJECT_ROOT,
            env=os.environ | {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        atomic_json(state, {
            "status": "running", "started": started, "runner_pid": runner.pid,
            "config": str(config_path), "runner_log": str(runner_log), "health_checks": str(health)})
        start, delay_index = monotonic(), 0
        with health.open("a") as health_file:
            while runner.poll() is None:
                delay = CHECK_DELAYS[delay_index] if delay_index < len(CHECK_DELAYS) else ROUTINE_DELAY
                delay_index += 1
                try:
                    runner.wait(timeout=delay)
                except subprocess.TimeoutExpired:
                    pass
                health_file.write(json.dumps(snapshot(
                    config_path, runner, round(monotonic() - start), runner_log)) + "\n")
                health_file.flush()
    status = "complete" if runner.returncode == 0 else "failed"
    atomic_json(state, {
        "status": status, "started": started, "finished": datetime.now().astimezone().isoformat(),
        "runner_pid": runner.pid, "returncode": runner.returncode,
        "config": str(config_path), "runner_log": str(runner_log), "health_checks": str(health)})
    atomic_json(control / "finished.flag", {
        "finished": True, "status": status, "timestamp": datetime.now().astimezone().isoformat(),
        "returncode": runner.returncode, "completed_repeats": len(tuple(
            config.output_root.glob("cases/repeat_*/complete.json"))),
        "expected_repeats": config.training_repeats,
    })
    return runner.returncode or 0


def watch_finish(config_path: Path) -> int:
    """Add the explicit finish flag to a supervisor that was already running."""
    config = load_config(config_path)
    control = config.output_root.parent / f"{config.output_root.name}_control"
    state_path = control / "supervisor.json"
    while True:
        state = json.loads(state_path.read_text())
        if state["status"] in {"complete", "failed"}:
            atomic_json(control / "finished.flag", {
                "finished": True, "status": state["status"], "timestamp": state["finished"],
                "returncode": state["returncode"], "completed_repeats": len(tuple(
                    config.output_root.glob("cases/repeat_*/complete.json"))),
                "expected_repeats": config.training_repeats,
            })
            return 0
        sleep(60)


def launch(config_path: Path) -> str:
    """Start the supervisor in an independent transient user service."""
    config_path = config_path.resolve()
    config = load_config(config_path)
    control = config.output_root.parent / f"{config.output_root.name}_control"
    control.mkdir(parents=True, exist_ok=True)
    unit = "fl-wz-" + re.sub(r"[^a-z0-9]+", "-", config.name.lower()).strip("-")[:48]
    command = (
        "systemd-run", "--user", "--collect", f"--unit={unit}", "--service-type=exec",
        f"--working-directory={PROJECT_ROOT}", "--property=OOMPolicy=continue",
        "--property=KillMode=control-group", sys.executable, "-u", "-m",
        "Experiments.warm_start_gradient_retraining.supervisor", "supervise", str(config_path))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    atomic_json(control / "launch.json", {
        "timestamp": datetime.now().astimezone().isoformat(), "unit": unit,
        "config": str(config_path), "command": list(command), "systemd": result.stdout.strip()})
    return unit


def main() -> None:
    """Launch or internally supervise one config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "supervise", "watch-finish"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    if args.action == "launch":
        print(launch(args.config))
    elif args.action == "supervise":
        raise SystemExit(supervise(args.config.resolve()))
    else:
        raise SystemExit(watch_finish(args.config.resolve()))


if __name__ == "__main__":
    main()
