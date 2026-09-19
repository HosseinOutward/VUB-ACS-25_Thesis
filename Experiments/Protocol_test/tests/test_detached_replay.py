from __future__ import annotations

from itertools import islice
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from Experiments.Protocol_test import detached_replay
from Experiments.Protocol_test.detached_replay import check_delays


def test_health_check_schedule_is_1_5_30_then_every_90_minutes() -> None:
    delays = tuple(islice(check_delays(), 5))
    assert delays == (60, 240, 1500, 5400, 5400)
    assert tuple(sum(delays[:index]) // 60 for index in range(1, 6)) == (1, 5, 30, 120, 210)


def test_health_probe_failures_are_recorded_without_aborting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(command: tuple[str, ...], **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(detached_replay.subprocess, "run", fail)
    process = SimpleNamespace(pid=123, poll=lambda: None)
    plan = SimpleNamespace(output_root=tmp_path, cases=(object(),))

    snapshot = detached_replay._health_snapshot(
        plan, process, process, elapsed_seconds=60, runner_log=tmp_path / "runner.log",
    )

    assert snapshot["process_tree_rss_mib"] is None
    assert snapshot["gpu_status"] is None
    assert set(snapshot["probe_errors"]) == {"process_tree_rss_mib", "gpu_status"}


def test_plotter_grace_covers_in_flight_and_final_renders() -> None:
    assert detached_replay._PLOTTER_SHUTDOWN_GRACE_SECONDS > (
        2 * detached_replay.RENDER_TIMEOUT_SECONDS + detached_replay.REFRESH_SECONDS
    )
