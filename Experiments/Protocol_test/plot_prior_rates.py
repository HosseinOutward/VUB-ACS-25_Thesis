"""Retrained-prior rate per round for the frozen-reuse windows.

The prior is the rate a working conditional entropy coder would reach, so this is the side
information's value before any transport touches it.  `marginal_rate` is the same quantizer's
rate with no side information at all, and is drawn as the reference each family is measured
against.  Rounds are averaged over the clients that have reported them; nothing else is derived.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["r56_femnist_nat", "r56_femnist", "r56_svhn", "r56_cifar10"]
FAMILY = {"T": ("T  (own past 5)", "#1B6AA8"), "O": ("O  (30 clients x 5)", "#C1440E")}
STYLE = {5: "-", 12: "--"}


def rounds_of(csv: Path, column: str, clients: int) -> list[tuple[int, float]]:
    """Per-round mean of one rate column, over rounds every client has reported."""
    frame = pd.read_csv(csv)
    whole = frame.groupby("round_id").filter(lambda g: len(g) == clients)
    return sorted((int(r), g[column].mean()) for r, g in whole.groupby("round_id"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path,
                        default=Path("Experiments/Protocol_test/exp_records/binary_frozen"))
    parser.add_argument("--out", type=Path,
                        default=Path("Experiments/Protocol_test/analysis_plots/binary_frozen/prior_rates.png"))
    parser.add_argument("--clients", type=int, default=30)
    args = parser.parse_args()

    combos = [c for c in ORDER if (args.records / c).exists()]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), squeeze=False)
    for ax, combo in zip(axes.ravel(), combos):
        for codec, (label, color) in FAMILY.items():
            for csv in sorted((args.records / combo / "cases" / codec).glob("*/compression_records.csv")):
                prior = rounds_of(csv, "prior_rate", args.clients)
                if not prior:
                    continue
                start = prior[0][0]
                ax.plot([r for r, _ in prior], [v for _, v in prior], STYLE.get(start, ":"),
                        color=color, lw=1.6, label=f"{label}, trained @{start}")
                ax.plot(start, prior[0][1], "o", color=color, ms=7, mfc="white", mew=1.8)
                marginal = rounds_of(csv, "marginal_rate", args.clients)
                ax.plot([r for r, _ in marginal], [v for _, v in marginal], ":",
                        color="#8A8F98", lw=1.1,
                        label="no side information" if codec == "T" and start == 5 else None)
        ax.set_title(combo, fontsize=10.5)
        ax.set_xlabel("round", fontsize=9)
        ax.set_ylabel("retrained prior rate (bit/coordinate)", fontsize=9)
        ax.grid(alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(combos):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle("open marker = the training round; every later round reuses that quantizer",
                 y=.925, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, .89])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)

    print(f"-> {args.out}\n")
    print(f"{'combo':17s} {'window':9s} {'T prior':>8s} {'O prior':>8s} {'marginal':>9s} "
          f"{'T buys':>7s} {'O buys':>7s}")
    for combo in combos:
        for start in (5, 12):
            got = {}
            for codec in FAMILY:
                for csv in (args.records / combo / "cases" / codec).glob(f"round{start}_*/compression_records.csv"):
                    p = rounds_of(csv, "prior_rate", args.clients)
                    m = rounds_of(csv, "marginal_rate", args.clients)
                    if p:
                        got[codec] = (sum(v for _, v in p) / len(p), sum(v for _, v in m) / len(m))
            if len(got) == 2:
                (t, mt), (o, mo) = got["T"], got["O"]
                print(f"{combo:17s} r{start}..{start+12:<3d} {t:8.4f} {o:8.4f} {(mt+mo)/2:9.4f} "
                      f"{100*(1-t/mt):6.1f}% {100*(1-o/mo):6.1f}%")


if __name__ == "__main__":
    main()
