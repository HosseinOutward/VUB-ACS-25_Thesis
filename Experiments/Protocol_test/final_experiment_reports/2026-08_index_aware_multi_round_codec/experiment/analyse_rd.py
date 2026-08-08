from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

import pandas as pd


REPORT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.Protocol_test.analysis import averaged_curve
from FL_code.FL_core.rd_metrics import rd_curve_gap


MODES = ("O", "T", "OM", "TM", "OMI", "TMI")
PAIRS = (
    ("O", "T"),
    ("OM", "O"),
    ("OM", "T"),
    ("TM", "T"),
    ("OMI", "O"),
    ("OMI", "OM"),
    ("OMI", "T"),
    ("TMI", "T"),
    ("TMI", "TM"),
    ("OMI", "TMI"),
)


def read_records() -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Load the six final curves and require identical five-client, nine-round coverage."""
    paths = {
        mode: REPORT_ROOT / f"results/records/{mode}/compression_records.csv"
        for mode in MODES
    }
    frames = {mode: pd.read_csv(path) for mode, path in paths.items()}
    expected = {(round_id, client_id) for round_id in range(12, 21) for client_id in range(5)}
    for mode, frame in frames.items():
        actual = set(zip(frame["round_id"], frame["client_id"], strict=True))
        assert actual == expected and len(frame) == len(expected), (
            f"{mode} coverage differs from the expected 45 rows: {actual ^ expected}."
        )
    return paths, frames


def write_rd_gaps(
    paths: dict[str, Path],
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write matched-distortion and matched-rate gaps for training, frozen, and full curves."""
    selectors: tuple[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]], ...] = (
        ("training", lambda frame: frame[frame["round_id"].eq(12)]),
        ("frozen", lambda frame: frame[frame["round_id"].ne(12)]),
        ("full", lambda frame: frame),
    )
    rows: list[dict[str, float | int | str]] = []
    curve_rows: list[dict[str, float | int | str]] = []
    for subset, select in selectors:
        selected = {mode: select(frame) for mode, frame in frames.items()}
        curves = {
            mode: averaged_curve(frame, paths[mode])
            for mode, frame in selected.items()
        }
        for candidate, reference in PAIRS:
            rows.append({
                "subset": subset,
                "candidate": candidate,
                "reference": reference,
                "rows": len(selected[candidate]),
                **rd_curve_gap(
                    *curves[candidate],
                    *curves[reference],
                    name=f"{candidate} vs {reference} ({subset})",
                ),
            })
        for mode, (rates, distortions) in curves.items():
            curve_rows.extend(
                {
                    "subset": subset,
                    "mode": mode,
                    "stage": stage,
                    "cumulative_prior_rate": rate,
                    "wmape_db": distortion,
                }
                for stage, (rate, distortion) in enumerate(
                    zip(rates, distortions, strict=True), start=1
                )
            )
    pd.DataFrame(rows).to_csv(output_dir / "rd_gaps.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output_dir / "rd_curves.csv", index=False)


def write_baseline_reproduction(
    paths: dict[str, Path],
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Compare the replacement O/T baselines with the records they superseded."""
    previous = {
        "O": REPORT_ROOT / "results/archive/baselines/original_run_O_previous/compression_records.csv",
        "T": REPORT_ROOT / "results/archive/baselines/original_run_t_previous/compression_records.csv",
    }
    rows = []
    for mode, previous_path in previous.items():
        prior = pd.read_csv(previous_path)
        rows.append({
            "candidate": mode,
            "reference": f"previous_{mode}",
            "rows": len(frames[mode]),
            **rd_curve_gap(
                *averaged_curve(frames[mode], paths[mode]),
                *averaged_curve(prior, previous_path),
                name=f"new {mode} vs previous {mode}",
            ),
        })
    pd.DataFrame(rows).to_csv(output_dir / "baseline_reproduction.csv", index=False)


def write_archive_gaps(output_dir: Path) -> None:
    """Summarize every retained indexed probe against its matched historical O/T rows."""
    archive = REPORT_ROOT / "results/archive"
    references = {
        "O": archive / "baselines/original_run_O_previous/compression_records.csv",
        "T": archive / "baselines/original_run_t_previous/compression_records.csv",
    }
    reference_frames = {name: pd.read_csv(path) for name, path in references.items()}
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(archive.rglob("compression_records.csv")):
        if "baselines" in path.parts or path.parent.name == "baseline_original_O":
            continue
        mode = path.parent.name.removeprefix("index_aware_")
        reference = "O" if mode in {"OI", "OMI"} else "T"
        candidate = pd.read_csv(path)
        keys = set(zip(candidate["round_id"], candidate["client_id"], strict=True))
        baseline = reference_frames[reference]
        baseline = baseline[
            [key in keys for key in zip(baseline["round_id"], baseline["client_id"], strict=True)]
        ]
        assert len(candidate) == len(baseline), f"No matched baseline rows for {path}."
        for subset, selector in (
            ("training", candidate["round_id"].eq(12)),
            ("frozen", candidate["round_id"].ne(12)),
            ("available", pd.Series(True, index=candidate.index)),
        ):
            selected = candidate[selector]
            if selected.empty:
                continue
            selected_keys = set(zip(selected["round_id"], selected["client_id"], strict=True))
            matched = baseline[
                [
                    key in selected_keys
                    for key in zip(baseline["round_id"], baseline["client_id"], strict=True)
                ]
            ]
            rows.append({
                "evidence": path.parent.relative_to(archive).as_posix(),
                "mode": mode,
                "reference": reference,
                "subset": subset,
                "rows": len(selected),
                **rd_curve_gap(
                    *averaged_curve(selected, path),
                    *averaged_curve(matched, references[reference]),
                    name=f"{path.parent.name} vs {reference} ({subset})",
                ),
            })
    pd.DataFrame(rows).to_csv(output_dir / "archive_rd_gaps.csv", index=False)


def main() -> None:
    """Regenerate the compact RD evidence consumed by the investigation report."""
    paths, frames = read_records()
    output_dir = REPORT_ROOT / "results/analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rd_gaps(paths, frames, output_dir)
    write_baseline_reproduction(paths, frames, output_dir)
    write_archive_gaps(output_dir)


if __name__ == "__main__":
    main()
