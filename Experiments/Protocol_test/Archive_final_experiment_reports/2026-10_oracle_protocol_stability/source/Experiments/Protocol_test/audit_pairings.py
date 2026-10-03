"""Compare preserved protocol RD curves with explicit matched coverage and outlier exclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Experiments.Protocol_test.rd_metrics import rd_curve_gap, wmape_to_db
from Experiments.Protocol_test.records import read_available_case


def audit(container: Path, output: Path) -> None:
    """Write full stage tables, equal-distortion gaps, and reproducible post-hoc exclusions."""
    output.mkdir(parents=True, exist_ok=True)
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    suspects: list[dict[str, object]] = []
    for label, codec, suffix in (
        ("OR", "OR", "warm"), ("TR", "TR", "warm"),
        ("OMR", "OMR", "w503020_gate"), ("TMR", "TMR", "w503020_gate"),
        ("OMR legacy", "OMR", "warm"), ("TMR legacy", "TMR", "warm"),
    ):
        for case in sorted((container / f"{codec.lower()}_svhn_30c_f9_{suffix}/runs/cases/{codec}").glob("start*")):
            identity = json.loads((case / "pre_run_identity.json").read_text())["identity"]
            frame = read_available_case(case, identity)
            start = int(case.name.removeprefix("start"))
            frames[label, start] = frame
            for row in frame.to_dict("records"):
                for source in ("coder_stage_rates", "server_si_rans_stage_rates"):
                    for stage, rate in enumerate(json.loads(row[source]), 1):
                        if rate < .01 or rate > 1.05:
                            suspects.append({"codec": label, "start": start, "round": row["round_id"],
                                "client": row["client_id"], "source": source, "stage": stage,
                                "rate": rate, "status": "dead" if rate < .01 else "blown"})
    requests = [("OR vs TR", "OR", "TR", start, start, False) for start in (5, 12, 20)]
    requests += [
        ("OR vs OMR", "OR", "OMR", 12, 12, False),
        ("OMR vs TMR revised", "OMR", "TMR", 7, 7, False),
        ("OMR vs TMR legacy", "OMR legacy", "TMR legacy", 6, 6, True),
        ("OR vs OMR frozen different placements", "OR", "OMR", 5, 7, True),
    ]
    points: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for pairing, left, right, left_start, right_start, complete_rounds in requests:
        pair = [frames[label, start] for label, start in ((left, left_start), (right, right_start))]
        keys = set(zip(pair[0].round_id, pair[0].client_id)) & set(zip(pair[1].round_id, pair[1].client_id))
        if complete_rounds:
            rounds = set.intersection(*(set(f.groupby("round_id").size().loc[lambda x: x == 30].index) for f in pair))
            keys = {key for key in keys if key[0] in rounds}
        if left_start != right_start:
            keys = {key for key in keys if key[0] > max(left_start, right_start)}
        outliers = {int(row["client"]) for row in suspects
                    if (row["codec"], row["start"]) in ((left, left_start), (right, right_start))}
        present_outliers = outliers & {client for _, client in keys}
        exclusions.append({"pairing": pairing, "start": left_start, "reference_start": right_start,
            "observed_outliers": sorted(outliers), "excluded_matched_clients": sorted(present_outliers)})
        for population in ("all clients", "exclude outliers"):
            selected = keys if population == "all clients" else {key for key in keys if key[1] not in outliers}
            matched = [f[[key in selected for key in zip(f.round_id, f.client_id)]] for f in pair]
            for phase in ("overall", "trained", "frozen"):
                groups = [f if phase == "overall" else f[f.round_id.eq(start) if phase == "trained" else f.round_id.gt(start)]
                          for f, start in zip(matched, (left_start, right_start), strict=True)]
                if any(group.empty for group in groups):
                    continue
                assert len(groups[0]) == len(groups[1])
                for label, group in zip((left, right), groups, strict=True):
                    wmape = np.array([json.loads(value) for value in group.stage_wmapes]).mean(axis=0)
                    rates = {}
                    for source, field in (
                        ("pre_rate", "quantizer_prior_stage_rates"), ("transport_rans", "coder_stage_rates"),
                        ("full_rans", "server_si_rans_stage_rates"), ("temporal_rans", "temporal_si_rans_stage_rates"),
                    ):
                        rates[source] = (np.cumsum(np.array([json.loads(value) for value in group[field]]).mean(axis=0))
                                         if field in group and group[field].notna().all() else np.full(3, np.nan))
                    for stage in range(3):
                        points.append({"pairing": pairing, "start": left_start, "reference_start": right_start,
                            "population": population, "phase": phase, "codec": label, "stage": stage + 1,
                            "rows": len(group), "clients": group.client_id.nunique(),
                            "rounds": sorted(group.round_id.unique().tolist()),
                            **{source: values[stage] for source, values in rates.items()},
                            "temporal_prior": group.temporal_si_prior_rate.mean()
                                if stage == 2 and "temporal_si_prior_rate" in group else np.nan,
                            "full_prior": group.server_si_prior_rate.mean() if stage == 2 else np.nan,
                            "wmape": wmape[stage], "wmape_db": wmape_to_db(wmape)[stage]})
    table = pd.DataFrame(points)
    gaps: list[dict[str, object]] = []
    group_fields = ["pairing", "start", "reference_start", "population", "phase"]
    for key, group in table.groupby(group_fields, sort=False):
        left, right = group.codec.unique()
        a, b = (group[group.codec == label].sort_values("stage") for label in (left, right))
        for source in ("transport_rans", "full_rans", "pre_rate"):
            gaps.append({**dict(zip(group_fields, key, strict=True)), "source": source, "rows": int(a.rows.iloc[0]),
                **rd_curve_gap(a[source].to_numpy(), a.wmape_db.to_numpy(), b[source].to_numpy(), b.wmape_db.to_numpy())})
    # Starts are pooled by their retained observation counts, without changing paired populations.
    pooled = table[table.pairing.eq("OR vs TR")].copy()
    for (population, phase, codec, stage), group in pooled.groupby(["population", "phase", "codec", "stage"]):
        row = group.iloc[0].to_dict()
        row.update(start="pooled", reference_start="pooled", rows=int(group.rows.sum()))
        for field in ("pre_rate", "transport_rans", "full_rans", "temporal_rans", "temporal_prior", "full_prior", "wmape"):
            row[field] = np.average(group[field], weights=group.rows)
        row["wmape_db"] = float(wmape_to_db(np.array([row["wmape"]]))[0])
        points.append(row)
    table = pd.DataFrame(points)
    for (population, phase), group in table[table.start.eq("pooled")].groupby(["population", "phase"]):
        a, b = (group[group.codec == label].sort_values("stage") for label in ("OR", "TR"))
        for source in ("transport_rans", "full_rans", "pre_rate"):
            gaps.append({"pairing": "OR vs TR", "start": "pooled", "reference_start": "pooled",
                "population": population, "phase": phase, "source": source, "rows": int(a.rows.iloc[0]),
                **rd_curve_gap(a[source].to_numpy(), a.wmape_db.to_numpy(), b[source].to_numpy(), b.wmape_db.to_numpy())})
    table.to_csv(output / "stage_points.csv", index=False)
    pd.DataFrame(gaps).to_csv(output / "comparison_gaps.csv", index=False)
    pd.DataFrame(suspects).to_csv(output / "suspect_stages.csv", index=False)
    pd.DataFrame(exclusions).to_csv(output / "exclusions.csv", index=False)
    lines = ["# Matched protocol RD and outlier sensitivity", "",
        "Negative gaps favor the first codec. Rates are cumulative bits/float; WMAPE is averaged in linear space before dB conversion.",
        "Outliers: any native or full-SI rANS stage <0.01 or >1.05 bits anywhere in that case's observed window.",
        "Exclude the union of flagged clients from BOTH codecs at the same start, for all matched rounds. This is post-hoc sensitivity analysis, not a deployable selection policy.",
        "The legacy OMR/TMR comparison retains only complete matched rounds 6–12. Revised OMR/TMR has only two trained clients.",
        "OR/OMR has a shared start only at 12 (14 trained clients). The supplementary frozen comparison uses OR start5 versus OMR start7, rounds 8–14: different training placement/history, not a controlled ablation.",
        "Fine-tuned prior stage breakdowns are unavailable: only final totals are shown, so no equal-distortion fine-tuned-prior curve gap is inferred.", "",
        "## Exclusions", "", "| Pairing | Starts | Observed outliers | Removed matched clients |", "|---|---|---|---|"]
    for row in exclusions:
        lines.append(f"| {row['pairing']} | {row['start']}/{row['reference_start']} | {row['observed_outliers']} | {row['excluded_matched_clients']} |")
    lines += ["", "## Equal-distortion gaps", "", "| Pairing | Start | Population | Phase | Rows/codec | Source | Gap bits/float | Gap % |", "|---|---|---|---|---:|---|---:|---:|"]
    for row in gaps:
        lines.append(f"| {row['pairing']} | {row['start']} | {row['population']} | {row['phase']} | {row['rows']} | {row['source']} | {row['rate_gap_bits']:+.5f} | {row['rate_gap_percent']:+.2f}% |")
    for key, group in table.groupby(group_fields, sort=False):
        lines += ["", "## " + " / ".join(map(str, key)), "",
            f"Rows per codec: {group.rows.iloc[0]}. Phases with no matched coverage are not reported.", "",
            "| Codec | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |",
            "|---|---:|---:|---:|---:|---:|---:|---|"]
        for row in group.to_dict("records"):
            values = " | ".join("—" if pd.isna(row[field]) else f"{row[field]:.4f}" for field in
                                ("pre_rate", "temporal_prior", "full_prior", "temporal_rans", "full_rans"))
            lines.append(f"| {row['codec']} | {row['stage']} | {values} | {row['wmape_db']:.3f} ({row['wmape']:.3f}%) |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(pd.DataFrame(gaps).to_string(index=False))


def main() -> None:
    """Analyze the preserved datasets without editing or restarting any experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.container, args.output)


if __name__ == "__main__":
    main()
