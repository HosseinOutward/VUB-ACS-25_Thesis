"""Launch and supervise the full warm-start study as a detached user service."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic
from typing import Any

from Experiments.warm_start_retraining.experiment import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECK_DELAYS = (60, 4 * 60, 25 * 60)
ROUTINE_DELAY = 90 * 60


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON control artifact."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def snapshot(config_path: Path, runner: subprocess.Popen[bytes], plotter: subprocess.Popen[bytes],
             elapsed: int, runner_log: Path) -> dict[str, Any]:
    """Capture scheduled process, output, memory, GPU, and error health."""
    config = load_config(config_path)
    text = runner_log.read_text(errors="replace") if runner_log.exists() else ""
    mem = {
        fields[0].rstrip(":"): int(fields[1]) // 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if (fields := line.split())[0].rstrip(":") in {"MemAvailable", "SwapFree"}
    }
    gpu = subprocess.run(
        ("nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu",
         "--format=csv,noheader"), capture_output=True, text=True,
    ).stdout.splitlines()
    return {
        "timestamp": datetime.now().astimezone().isoformat(), "elapsed_seconds": elapsed,
        "runner_pid": runner.pid, "runner_alive": runner.poll() is None,
        "returncode": runner.poll(), "plotter_pid": plotter.pid,
        "plotter_alive": plotter.poll() is None,
        "completed_cases": len(tuple(config.output_root.glob("cases/*/repeat_*/complete.json"))),
        "expected_cases": len(config.architectures) * config.repeats,
        "error_markers": [marker for marker in ("Traceback", "out of memory", "CUDA error")
                          if marker.lower() in text.lower()],
        **mem, "gpu_status": gpu,
    }


def supervise(config_path: Path) -> int:
    """Run the experiment and live analyser with repository-standard health intervals."""
    config = load_config(config_path)
    control = config.output_root.parent / f"{config.output_root.name}_control"
    control.mkdir(parents=True, exist_ok=True)
    runner_log = control / "runner.log"
    plotter_log = control / "plot_refresh.log"
    health = control / "health_checks.jsonl"
    state = control / "supervisor.json"
    started = datetime.now().astimezone().isoformat()
    with runner_log.open("ab", buffering=0) as output, plotter_log.open("ab", buffering=0) as plot_output:
        runner = subprocess.Popen(
            (sys.executable, "-u", "-m", "Experiments.warm_start_retraining.experiment", str(config_path)),
            cwd=PROJECT_ROOT, env=os.environ | {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
        )
        plotter = subprocess.Popen(
            (sys.executable, "-u", "-m", "Experiments.warm_start_retraining.live_plot",
             str(config_path), "--runner-pid", str(runner.pid)),
            cwd=PROJECT_ROOT, env=os.environ | {"PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL, stdout=plot_output, stderr=subprocess.STDOUT,
        )
        atomic_json(state, {
            "status": "running", "started": started, "runner_pid": runner.pid,
            "plotter_pid": plotter.pid, "config": str(config_path),
            "runner_log": str(runner_log), "plotter_log": str(plotter_log), "health_checks": str(health),
        })
        start = monotonic()
        delay_index = 0
        with health.open("a") as health_file:
            while runner.poll() is None:
                delay = CHECK_DELAYS[delay_index] if delay_index < len(CHECK_DELAYS) else ROUTINE_DELAY
                delay_index += 1
                try:
                    runner.wait(timeout=delay)
                except subprocess.TimeoutExpired:
                    pass
                health_file.write(json.dumps(snapshot(
                    config_path, runner, plotter, round(monotonic() - start), runner_log)) + "\n")
                health_file.flush()
        plotter.wait(timeout=360)
    status = "complete" if runner.returncode == 0 else "failed"
    atomic_json(state, {
        "status": status, "started": started, "finished": datetime.now().astimezone().isoformat(),
        "runner_pid": runner.pid, "returncode": runner.returncode, "plotter_pid": plotter.pid,
        "plotter_returncode": plotter.returncode, "config": str(config_path),
        "runner_log": str(runner_log), "plotter_log": str(plotter_log), "health_checks": str(health),
    })
    return runner.returncode or 0


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
        "Experiments.warm_start_retraining.supervisor", "supervise", str(config_path),
    )
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    atomic_json(control / "launch.json", {
        "timestamp": datetime.now().astimezone().isoformat(), "unit": unit,
        "config": str(config_path), "command": list(command), "systemd": result.stdout.strip(),
    })
    return unit


def main() -> None:
    """Launch or internally supervise one warm-start config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "supervise"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    if args.action == "launch":
        print(launch(args.config))
    else:
        raise SystemExit(supervise(args.config.resolve()))


if __name__ == "__main__":
    main()
