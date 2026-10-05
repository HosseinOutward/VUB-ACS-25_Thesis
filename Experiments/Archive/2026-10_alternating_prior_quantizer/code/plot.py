"""Report paired equal-distortion phase matrices and stage RD points for alternating fits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from Experiments.Protocol_test.rd_metrics import rd_curve_gap, wmape_to_db


def phase_curve(
    records: list[dict[str, Any]], phase: int, clients: set[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Average cumulative rates and raw WMAPE over exactly the same clients at every stage."""
    rows = [row for row in records if row["phase_index"] == phase and row["client_id"] in clients]
    stages = sorted({row["stage"] for row in rows})
    rates, wmape, marginal = (np.array([
        np.mean([row[field] for row in rows if row["stage"] == stage]) for stage in stages
    ], dtype=np.float64) for field in ("prior_bits", "wmape_percent", "marginal_bits"))
    return rates, wmape, marginal


def gap(
    records: list[dict[str, Any]], row_phase: int, column_phase: int, clients: set[int],
) -> float | None:
    """Compute the repository's directional rate gap, leaving invalid RD curves unavailable."""
    if not clients:
        return None
    rate, wmape, _ = phase_curve(records, row_phase, clients)
    ref_rate, ref_wmape, _ = phase_curve(records, column_phase, clients)
    for rates, distortions in ((rate, wmape), (ref_rate, ref_wmape)):
        if (len(rates) < 2 or not np.isfinite([rates, distortions]).all() or (distortions <= 0).any()
                or not (np.diff(rates) > 0).all() or not (np.diff(distortions) < 0).all()):
            return None
    value = rd_curve_gap(rate, wmape_to_db(wmape), ref_rate, wmape_to_db(ref_wmape))["rate_gap_percent"]
    return float(value) if np.isfinite(value) else None


def render(root: Path, output: Path) -> None:
    """Regenerate phase comparisons and plots from the archived measurement table."""
    config = json.loads((root / "config.json").read_text())["identity"]
    phases = config["plan"]["phases"]
    labels = [f"{index}:{phase}" for index, phase in enumerate(phases)]
    stage_count = config["quantizer"]["num_planes"]
    records: list[dict[str, Any]] = []
    with (root / "measurements.csv").open() as stream:
        for row in csv.DictReader(stream):
            records.append({key: int(value) if key in {"client_id", "phase_index", "stage", "symbols"}
                            else value if key in {"phase", "flag"} else float(value)
                            for key, value in row.items()})
    output.mkdir(parents=True, exist_ok=True)
    clients_by_phase = {
        index: {client for client in config["plan"]["cases"]
                if {row["stage"] for row in records if row["phase_index"] == index and row["client_id"] == client}
                == set(range(1, stage_count + 1))}
        for index in range(len(phases))
    }
    flagged = {row["client_id"] for row in records if row["flag"]}
    complete = all(clients == set(config["plan"]["cases"]) for clients in clients_by_phase.values())
    report = [
        f"# Alternating training diagnostic — {'complete' if complete else 'PARTIAL'}\n",
        "Single trained target round; these are successive fits on the same data, not a trained+9-frozen protocol window.\n",
        f"Seed {config['plan']['seed']}; target round {config['plan']['target_round']}; "
        f"fixed full decoder SI from rounds {config['plan']['target_round'] - 2}–{config['plan']['target_round'] - 1}, "
        "all five clients (10 compressed-history vectors). Raw prefix is an explicitly available decoder seed.\n",
        "Each prior phase refits fixed hard symbols with the existing best-of-three prior fitter. "
        "Quantizer/joint phases each continue one trajectory with the existing warm schedule and RD objective. "
        "Rates are hard-symbol conditional cross-entropies in bits/float; no entropy coder or symbol transport is run.\n",
        "Matrices: row versus column, negative favors the row. Both directions are recomputed. "
        "Clients are matched per pair; rates and raw WMAPE are averaged before dB conversion and interpolation. "
        "Invalid/nonmonotone/nonoverlapping curves are —.\n",
        "| Phase | Clients measured |\n|---|---:|",
        *[f"| {label} | {len(clients_by_phase[index])}/5 |" for index, label in enumerate(labels)],
        "",
    ]
    matrix_rows: list[dict[str, Any]] = []
    for excluded in (False, True):
        report.extend([f"\n## {'With symmetric exclusions' if excluded else 'All clients'}\n",
                       "| Row / column | " + " | ".join(labels) + " |",
                       "|---|" + "---:|" * len(labels)])
        for row_phase, label in enumerate(labels):
            cells: list[str] = []
            for column_phase in range(len(phases)):
                clients = clients_by_phase[row_phase] & clients_by_phase[column_phase]
                if excluded:
                    clients -= flagged
                value = gap(records, row_phase, column_phase, clients)
                cells.append("—" if value is None else f"{value:+.2f}%")
                matrix_rows.append({
                    "exclusions": excluded, "row": label, "column": labels[column_phase],
                    "rate_gap_percent": value, "retained_clients": ",".join(map(str, sorted(clients))),
                    "coverage": f"{len(clients)}/5",
                })
            report.append(f"| {label} | " + " | ".join(cells) + " |")
        if excluded:
            reasons = sorted({f"client {row['client_id']}, {labels[row['phase_index']]}, stage {row['stage']}: {row['flag']}"
                              for row in records if row["flag"]})
            report.append("\nExcluded across the whole diagnostic: " + ("; ".join(reasons) or "none") + ".\n")
            report.append("Pair-specific retained client IDs and coverage are in `phase_gaps.csv`.\n")
    report.extend([
        "\n## Stage RD points, all available clients\n",
        "Cumulative stage rates; all reported stages were measured. The current prior follows each phase's weights.\n",
        "| Phase | Stage | Clients | Marginal | Initial joint/pre-fit | Full-SI fitted prior | Current prior | Native rANS | Temporal prior/rANS | Full-SI rANS | WMAPE dB (raw %) |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ])
    points: list[dict[str, Any]] = []
    figure, axis = plt.subplots(figsize=(9, 6))
    for index, label in enumerate(labels):
        clients = clients_by_phase[index]
        if not clients:
            continue
        rates, wmape, marginal = phase_curve(records, index, clients)
        db = wmape_to_db(wmape) if np.isfinite(wmape).all() and (wmape > 0).all() else np.full_like(wmape, np.nan)
        axis.plot(db, rates, marker="o", label=f"{label} ({len(clients)}/5)")
        for stage, (rate, distortion, distortion_db, marginal_rate) in enumerate(zip(rates, wmape, db, marginal, strict=True), 1):
            points.append({"phase": label, "stage": stage, "clients": len(clients), "prior_bits": rate,
                           "marginal_bits": marginal_rate, "wmape_percent": distortion, "wmape_db": distortion_db})
            report.append(f"| {label} | {stage} | {len(clients)}/5 | {marginal_rate:.6f} | "
                          f"{f'{rate:.6f}' if phases[index] == 'initial_joint' else '—'} | "
                          f"{f'{rate:.6f}' if phases[index] == 'prior' else '—'} | {rate:.6f} | — | — | — | "
                          f"{distortion_db:.3f} ({distortion:.4f}%) |")
    axis.set(xlabel="WMAPE (dB)", ylabel="Cumulative conditional prior (bits/float)",
             title=f"Alternating training: target round {config['plan']['target_round']} — {'complete' if complete else 'PARTIAL'}")
    axis.grid(alpha=0.25)
    if points:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "rd_phases.png", dpi=160)
    plt.close(figure)
    for filename, rows in (("phase_gaps.csv", matrix_rows), ("stage_points.csv", points)):
        if rows:
            with (output / filename).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (output / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    """Regenerate the archived diagnostic into an explicitly selected output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    render(args.root, args.out or args.root / "analysis")


if __name__ == "__main__":
    main()
