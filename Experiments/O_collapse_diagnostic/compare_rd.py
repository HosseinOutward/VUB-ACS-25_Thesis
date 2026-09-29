"""Compare the RD curves of the candidates each configuration would select, per client and SI set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Experiments.Protocol_test.rd_metrics import rd_curve_gap, wmape_to_db

_KEY = ["client_id", "si_set"]


def load_candidates(folder: Path) -> pd.DataFrame:
    """Read every attempt with its cumulative fine-tuned rate curve and WMAPE-dB curve."""
    rows = [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]
    frame = pd.DataFrame(rows)
    frame["rate"] = frame.fine_tuned_stage_rates.map(np.cumsum)
    frame["wmape_db"] = frame.stage_wmapes.map(lambda values: wmape_to_db(np.array(values)))
    # The protocol keeps the lowest-loss attempt, so rank candidates the same way.
    frame["loss_rank"] = frame.groupby(_KEY).final_training_loss.rank(method="first").astype(int)
    return frame


def verdict(row: pd.Series) -> str:
    """Classify a candidate as usable, or name how it failed."""
    if min(row.fine_tuned_stage_rates) < 0.01:
        return "dead stage"
    if max(row.fine_tuned_stage_rates) > 2:
        return "prior blown"
    return f"ok {row.rate[-1]:.2f}b {row.wmape_db[-1]:+.1f}dB"


def main() -> None:
    """Print, per (client, SI set), each configuration's selected candidate and which one wins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    selected = pd.merge(
        *(load_candidates(folder).query("loss_rank == 1") for folder in (args.candidate, args.reference)),
        on=_KEY, how="outer", suffixes=("_cand", "_ref"))
    lines = []
    for _, row in selected.sort_values(_KEY).iterrows():
        cand, ref = (verdict(row.rename(lambda name, s=suffix: name.removesuffix(s)))
                     if isinstance(row[f"rate{suffix}"], np.ndarray) else "pending"
                     for suffix in ("_cand", "_ref"))
        if "pending" in (cand, ref):
            winner = "-"
        elif cand.startswith("ok") and ref.startswith("ok"):
            # Rate gap at equal distortion over the stages both curves span; fall back to distortion.
            gap = rd_curve_gap(row.rate_cand, row.wmape_db_cand, row.rate_ref, row.wmape_db_ref)
            rate_gap, db_gap = gap["rate_gap_percent"], gap["distortion_gap_db"]
            better = rate_gap < 0 if np.isfinite(rate_gap) else db_gap < 0
            winner = (f"{'candidate' if better else 'reference'} "
                      f"(rate {rate_gap:+.1f}% at equal distortion, {db_gap:+.2f} dB at equal rate)")
        elif cand.startswith("ok") or ref.startswith("ok"):
            winner = "candidate (reference unusable)" if cand.startswith("ok") else "reference (candidate unusable)"
        else:
            winner = "neither usable"
        lines.append({"client": row.client_id, "si_set": row.si_set,
                      args.reference.name: ref, args.candidate.name: cand, "winner": winner})
    pd.set_option("display.width", 220, "display.max_colwidth", 80)
    print(pd.DataFrame(lines).to_string(index=False))


if __name__ == "__main__":
    main()
