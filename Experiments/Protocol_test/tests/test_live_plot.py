from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from Experiments.Protocol_test import live_plot


def test_source_digest_changes_with_plot_source(tmp_path: Path) -> None:
    source = tmp_path / "plot.py"
    source.write_text("first")
    first = live_plot.source_digest((source,))
    source.write_text("second")
    assert live_plot.source_digest((source,)) != first


def test_result_digest_changes_when_records_grow(tmp_path: Path) -> None:
    records = tmp_path / "cases" / "T" / "start5" / "compression_records.csv"
    records.parent.mkdir(parents=True)
    records.write_text("header\n")
    first = live_plot.result_digest(tmp_path)
    records.write_text("header\nrow\n")
    assert live_plot.result_digest(tmp_path) != first


def test_failed_refresh_preserves_last_successful_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = tmp_path / "analysis"
    control = tmp_path / "control"
    analysis.mkdir()
    control.mkdir()
    (analysis / "plot.png").write_text("previous")
    monkeypatch.setattr(
        live_plot.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "temporary syntax error\n"),
    )

    assert not live_plot.refresh_once(tmp_path, control)
    assert (analysis / "plot.png").read_text() == "previous"
    assert not tuple(tmp_path.glob(".analysis-refresh-*"))


def test_successful_refresh_atomically_replaces_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = tmp_path / "analysis"
    control = tmp_path / "control"
    analysis.mkdir()
    control.mkdir()
    (analysis / "old.png").write_text("old")

    def render(command: tuple[str, ...], **kwargs) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--out") + 1])
        (output / "new.png").write_text("new")
        return subprocess.CompletedProcess(command, 0, "rendered\n")

    monkeypatch.setattr(live_plot.subprocess, "run", render)
    assert live_plot.refresh_once(tmp_path, control)
    assert (analysis / "new.png").read_text() == "new"
    assert not (analysis / "old.png").exists()
    assert not (tmp_path / ".analysis.previous").exists()


def test_watcher_retries_failed_refresh_without_input_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tmp_path / "cases" / "T" / "start5" / "compression_records.csv"
    records.parent.mkdir(parents=True)
    records.write_text("header\n")
    control = tmp_path / "control"
    attempts: list[bool] = []
    alive = iter((True, True, False))

    monkeypatch.setattr(live_plot, "process_start_ticks", lambda _pid: 1)
    monkeypatch.setattr(live_plot, "_runner_alive", lambda *_args: next(alive))
    monkeypatch.setattr(live_plot, "result_digest", lambda _root: "results")
    monkeypatch.setattr(live_plot, "source_digest", lambda: "sources")
    monkeypatch.setattr(live_plot, "sleep", lambda _seconds: None)

    def refresh(_output_root: Path, _control_dir: Path) -> bool:
        attempts.append(True)
        return len(attempts) > 1

    monkeypatch.setattr(live_plot, "refresh_once", refresh)

    assert live_plot.watch(tmp_path, 123, control) == 0
    assert len(attempts) == 2
