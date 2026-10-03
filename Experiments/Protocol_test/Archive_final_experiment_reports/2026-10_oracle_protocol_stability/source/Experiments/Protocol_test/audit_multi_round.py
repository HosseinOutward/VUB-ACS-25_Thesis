"""Derive matched-round OMR/TMR RD tables directly from preserved records, not stale plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Experiments.Protocol_test.rd_metrics import rd_curve_gap, wmape_to_db
from Experiments.Protocol_test.records import read_available_case


def audit(container: Path, output: Path) -> None:
    """Write all staged rates, matched-population gaps, and individual suspect stages."""
    output.mkdir(parents=True, exist_ok=True)
    frames = {}
    for codec in ("OMR", "TMR"):
        case = container / f"{codec.lower()}_svhn_30c_f9_warm/runs/cases/{codec}/start6"
        frames[codec] = read_available_case(case, json.loads((case / "pre_run_identity.json").read_text())["identity"])
        for field in ("temporal_si_rans_stage_rates", "temporal_si_prior_rate"):
            if field not in frames[codec]:
                frames[codec][field] = np.nan
    complete_rounds = set.intersection(*(set(frame.groupby("round_id").filter(lambda group: len(group) == 30).round_id)
                                         for frame in frames.values()))
    table, suspect, client_summary = [], [], []
    staged = {
        "pre_rate": "quantizer_prior_stage_rates", "temporal_rans": "temporal_si_rans_stage_rates",
        "full_rans": "server_si_rans_stage_rates", "transport_rans": "coder_stage_rates",
    }
    for codec, frame in frames.items():
        for row in frame.to_dict("records"):
            for source, field in staged.items():
                if not isinstance(row[field], str):
                    continue
                for stage, rate in enumerate(json.loads(row[field]), 1):
                    if rate < 0.01 or rate > 1.05:
                        suspect.append({"codec": codec, "round": row["round_id"], "client": row["client_id"],
                                        "phase": row["phase"], "source": source, "stage": stage, "rate": rate,
                                        "status": "dead" if rate < .01 else "blown"})
        matched = frame[frame.round_id.isin(complete_rounds)]
        for (client, phase), group in matched.groupby(["client_id", "phase"]):
            client_summary.append({"codec": codec, "client": client, "phase": phase, "rows": len(group),
                                   "pre_rate": group.quantizer_prior_rate.mean(),
                                   "temporal_prior": group.temporal_si_prior_rate.mean(),
                                   "full_prior": group.server_si_prior_rate.mean(),
                                   "full_rans": group.server_si_rans_rate.mean(), "wmape": group.wmape.mean()})
        for phase, group in (("all", matched), ("trained", matched[matched.phase == "trained"]),
                             ("frozen", matched[matched.phase == "frozen"])):
            distortion = np.array([json.loads(value) for value in group.stage_wmapes]).mean(axis=0)
            rates = {
                source: np.cumsum(np.array([json.loads(value) for value in group[field]]).mean(axis=0))
                if group[field].map(lambda value: isinstance(value, str)).all() else [np.nan] * 3
                for source, field in staged.items()
            }
            for stage in range(3):
                table.append({"codec": codec, "phase": phase, "stage": stage + 1, "rows": len(group),
                              "wmape": distortion[stage], "wmape_db": wmape_to_db(distortion)[stage],
                              **{source: values[stage] for source, values in rates.items()},
                              "temporal_prior": group.temporal_si_prior_rate.mean() if stage == 2 else np.nan,
                              "full_prior": group.server_si_prior_rate.mean() if stage == 2 else np.nan})
    points = pd.DataFrame(table)
    points.to_csv(output / "stage_points.csv", index=False)
    pd.DataFrame(suspect).to_csv(output / "suspect_stages.csv", index=False)
    pd.DataFrame(client_summary).to_csv(output / "client_summary.csv", index=False)
    gaps = []
    for phase, group in points.groupby("phase"):
        candidate, reference = (group[group.codec == codec].sort_values("stage") for codec in ("OMR", "TMR"))
        for source in ("pre_rate", "full_rans", "transport_rans"):
            gaps.append({"phase": phase, "source": source, **rd_curve_gap(candidate[source].to_numpy(), candidate.wmape_db.to_numpy(),
                                                                         reference[source].to_numpy(), reference.wmape_db.to_numpy())})
    pd.DataFrame(gaps).to_csv(output / "comparison_gaps.csv", index=False)
    lines = ["# Preserved OMR/TMR baseline (incomplete experiments)", "",
             f"Recorded rows: OMR {len(frames['OMR'])}/900; TMR {len(frames['TMR'])}/900.",
             f"Matched complete rounds: {sorted(complete_rounds)}; equal client weights, no exclusions.", "",
             "Rates are cumulative bits/float. Fine-tuned prior stage breakdowns were not recorded; only final-stage totals are available.",
             "OMR has no temporal-prior/rANS measurement in these original records; — means unavailable, not zero.", "",
             "| Phase | Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    def number(value: float) -> str:
        return "—" if pd.isna(value) else f"{value:.4f}"
    for row in table:
        lines.append(f"| {row['phase']} | {row['codec']} | {row['stage']} | " + " | ".join(number(row[field]) for field in
                     ("pre_rate", "temporal_prior", "full_prior", "temporal_rans", "full_rans")) +
                     f" | {row['wmape_db']:.3f} ({row['wmape']:.3f}%) |")
    lines += ["", "Equal-distortion mean rate gaps: OMR minus TMR; positive means OMR worse.", "",
              "| Phase | Rate source | Gap bits/float | Gap % |", "|---|---|---:|---:|"]
    for row in gaps:
        lines.append(f"| {row['phase']} | {row['source']} | {row['rate_gap_bits']:+.5f} | {row['rate_gap_percent']:+.3f}% |")
    lines += ["", "Suspect stage details: `suspect_stages.csv`. Thresholds are <0.01 and >1.05 bits per stage.",
              "These results alone cannot establish an oracle-vs-temporal limit: histories differ after decoding, selection is finite-budget, and frozen priors fail for specific clients."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    """Audit the existing start-6 experiment records without mutating them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.container, args.output)


if __name__ == "__main__":
    main()
