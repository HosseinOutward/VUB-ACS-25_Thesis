"""Launch sparse replay independently and record scheduled health checks until it exits."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic
from typing import Any, TypeVar

from Experiments.Protocol_test.live_plot import REFRESH_SECONDS, RENDER_TIMEOUT_SECONDS
from Experiments.Protocol_test.sparse_replay import ReplayPlan, load_plan


_REPO_ROOT = Path(__file__).resolve().parents[2]
_INITIAL_CHECK_DELAYS = (60, 4 * 60, 25 * 60)
_ROUTINE_CHECK_DELAY = 90 * 60
# The runner can exit during one render, after which changed final inputs require another render.
_PLOTTER_SHUTDOWN_GRACE_SECONDS = 2 * RENDER_TIMEOUT_SECONDS + REFRESH_SECONDS + 30
_ERROR_MARKERS = ("Traceback", "out of memory", "BrokenProcessPool", "CUDA error")
_ProbeResult = TypeVar("_ProbeResult")


def check_delays() -> Iterator[int]:
    """Yield waits producing checks at 1, 5, 30, then every 90 minutes."""
    yield from _INITIAL_CHECK_DELAYS
    while True:
        yield _ROUTINE_CHECK_DELAY


def _control_dir(plan: ReplayPlan) -> Path:
    return plan.output_root.parent / f"{plan.output_root.name}_control"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _unit_name(plan: ReplayPlan) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", plan.name.lower()).strip("-")
    return f"fl-wz-{normalized[:48]}"


def _meminfo_mib() -> dict[str, int]:
    wanted = {"MemAvailable", "SwapFree"}
    values = {
        fields[0].rstrip(":"): int(fields[1]) // 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if (fields := line.split())[0].rstrip(":") in wanted
    }
    return {key: values[key] for key in ("MemAvailable", "SwapFree")}


def _process_tree_rss_mib(root_pid: int) -> int:
    rows = subprocess.run(
        ("ps", "-eo", "pid=,ppid=,rss="), check=True, capture_output=True, text=True
    ).stdout.splitlines()
    processes = [tuple(map(int, row.split())) for row in rows]
    descendants = {root_pid}
    while children := {pid for pid, parent, _rss in processes if parent in descendants} - descendants:
        descendants.update(children)
    return sum(rss for pid, _parent, rss in processes if pid in descendants) // 1024


def _health_probe(
    name: str, probe: Callable[[], _ProbeResult], errors: dict[str, str],
) -> _ProbeResult | None:
    """Run auxiliary telemetry without allowing its failure to stop the experiment."""
    try:
        return probe()
    except (OSError, subprocess.SubprocessError, ValueError, LookupError) as error:
        errors[name] = f"{type(error).__name__}: {error}"
        return None


def _health_snapshot(
    plan: ReplayPlan, runner: subprocess.Popen[bytes], plotter: subprocess.Popen[bytes],
    elapsed_seconds: int, runner_log: Path,
) -> dict[str, Any]:
    case_root = plan.output_root / "cases"
    probe_errors: dict[str, str] = {}
    csvs = _health_probe(
        "record_paths", lambda: tuple(case_root.glob("*/*/compression_records.csv")), probe_errors,
    ) or ()
    log_text = _health_probe(
        "runner_log", lambda: runner_log.read_text(errors="replace"), probe_errors,
    ) if runner_log.is_file() else ""
    record_rows = _health_probe(
        "record_rows",
        lambda: {
            str(path.relative_to(plan.output_root)): max(0, len(path.read_text().splitlines()) - 1)
            for path in csvs
        },
        probe_errors,
    )
    completed_cases = _health_probe(
        "completed_cases", lambda: len(tuple(case_root.glob("*/*/complete.json"))), probe_errors,
    )
    process_tree_rss_mib = (
        _health_probe(
            "process_tree_rss_mib", lambda: _process_tree_rss_mib(runner.pid), probe_errors,
        )
        if runner.poll() is None else 0
    )
    memory: dict[str, int | None] = {"MemAvailable": None, "SwapFree": None} | (
        _health_probe("memory", _meminfo_mib, probe_errors) or {}
    )
    gpu_status = _health_probe(
        "gpu_status",
        lambda: subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
        probe_errors,
    )
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "runner_pid": runner.pid,
        "runner_alive": runner.poll() is None,
        "returncode": runner.poll(),
        "plotter_pid": plotter.pid,
        "plotter_alive": plotter.poll() is None,
        "plotter_returncode": plotter.poll(),
        "completed_cases": completed_cases,
        "expected_cases": len(plan.cases),
        "record_rows": record_rows,
        "error_markers": [
            marker for marker in _ERROR_MARKERS if marker.lower() in (log_text or "").lower()
        ],
        "process_tree_rss_mib": process_tree_rss_mib,
        **memory,
        "gpu_status": gpu_status,
        "probe_errors": probe_errors,
    }


def supervise(plan_path: Path, worker_module: str = "Experiments.Protocol_test.sparse_replay", plotter_module: str = "Experiments.Protocol_test.live_plot") -> int:
    """Run replay as a child and persist timed health snapshots until completion."""
    plan = load_plan(plan_path)
    control_dir = _control_dir(plan)
    control_dir.mkdir(parents=True, exist_ok=True)
    runner_log = control_dir / "runner.log"
    plotter_log = control_dir / "plot_refresh.log"
    health_path = control_dir / "health_checks.jsonl"
    state_path = control_dir / "supervisor.json"
    started = datetime.now().astimezone().isoformat()
    with runner_log.open("ab", buffering=0) as log, plotter_log.open("ab", buffering=0) as plot_log:
        runner = subprocess.Popen(
            (
                sys.executable,
                "-u",
                "-m",
                worker_module,
                str(plan_path),
            ),
            cwd=_REPO_ROOT,
            env=os.environ | {
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
            },
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        plotter = subprocess.Popen(
            (
                sys.executable, "-u", "-m", plotter_module,
                str(plan.output_root), "--runner-pid", str(runner.pid),
                "--control-dir", str(control_dir),
            ),
            cwd=_REPO_ROOT,
            env=os.environ | {"PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL,
            stdout=plot_log,
            stderr=subprocess.STDOUT,
        )
        _atomic_json(state_path, {
            "status": "running", "started": started, "runner_pid": runner.pid,
            "plotter_pid": plotter.pid,
            "plan": str(plan_path), "runner_log": str(runner_log),
            "plotter_log": str(plotter_log), "health_checks": str(health_path),
        })
        start = monotonic()
        with health_path.open("a") as health_log:
            for delay in check_delays():
                try:
                    runner.wait(timeout=delay)
                except subprocess.TimeoutExpired:
                    pass
                elapsed = round(monotonic() - start)
                health_log.write(json.dumps(
                    _health_snapshot(plan, runner, plotter, elapsed, runner_log)) + "\n")
                health_log.flush()
                if runner.poll() is not None:
                    break
        try:
            plotter.wait(timeout=_PLOTTER_SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            plotter.terminate()
            plotter.wait()
    status = "complete" if runner.returncode == 0 else "failed"
    _atomic_json(state_path, {
        "status": status, "started": started, "finished": datetime.now().astimezone().isoformat(),
        "runner_pid": runner.pid, "returncode": runner.returncode, "plan": str(plan_path),
        "plotter_pid": plotter.pid, "plotter_returncode": plotter.returncode,
        "runner_log": str(runner_log), "plotter_log": str(plotter_log),
        "health_checks": str(health_path),
    })
    return runner.returncode or 0


def launch(plan_path: Path, worker_module: str = "Experiments.Protocol_test.sparse_replay", plotter_module: str = "Experiments.Protocol_test.live_plot") -> str:
    """Start the supervisor in an independent transient user service."""
    plan_path = plan_path.resolve()
    plan = load_plan(plan_path)
    control_dir = _control_dir(plan)
    control_dir.mkdir(parents=True, exist_ok=True)
    unit = _unit_name(plan)
    command = (
        "systemd-run", "--user", "--collect", f"--unit={unit}", "--service-type=exec",
        f"--working-directory={_REPO_ROOT}", "--property=OOMPolicy=continue",
        "--property=KillMode=control-group", sys.executable, "-u", "-m",
        "Experiments.Protocol_test.detached_replay", "supervise", str(plan_path),
        "--worker-module", worker_module, "--plotter-module", plotter_module,
    )
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    _atomic_json(control_dir / "launch.json", {
        "timestamp": datetime.now().astimezone().isoformat(),
        "unit": unit,
        "plan": str(plan_path),
        "command": list(command),
        "systemd": result.stdout.strip(),
    })
    return unit


def main() -> None:
    """Launch or internally supervise one declared sparse replay plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "supervise"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--worker-module", default="Experiments.Protocol_test.sparse_replay")
    parser.add_argument("--plotter-module", default="Experiments.Protocol_test.live_plot")
    args = parser.parse_args()
    if args.action == "launch":
        print(launch(args.plan, args.worker_module, args.plotter_module), flush=True)
    else:
        raise SystemExit(supervise(args.plan.resolve(), args.worker_module, args.plotter_module))


if __name__ == "__main__":
    main()
