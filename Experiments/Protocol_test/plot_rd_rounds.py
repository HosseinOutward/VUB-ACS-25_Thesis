"""One dot per round on the rate-distortion plane, for every frozen-reuse window.

Nothing is averaged across rounds and nothing is derived: a dot is one codec at one round, over
the 30 clients that reported it.  Shape is the side-information family, fill is whether the rate
is the retrained prior (what a working conditional coder would reach) or the actual wire, and
colour is which round the quantizer was trained at.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["r56_femnist_nat", "r56_femnist", "r56_svhn", "r56_cifar10"]
# codec dir -> (family marker, rate column, filled?)
SERIES = {
    "T": ("o", "prior_rate", False),
    "O": ("^", "prior_rate", False),
    "T_rans_retrained": ("o", "coder_rate", True),
    "O_oracle_sw_retrained": ("^", "coder_rate", True),
}
COLOR = {5: "#1B6AA8", 12: "#C1440E"}


def dots(csv: Path, column: str, clients: int) -> list[tuple[int, float, float]]:
    """Per-round (round, rate, WMAPE dB) over rounds every client has reported."""
    frame = pd.read_csv(csv)
    whole = frame.groupby("round_id").filter(lambda g: len(g) == clients)
    return sorted(
        (int(r), g[column].mean(), 20 * np.log10(np.mean([eval(w)[0] for w in g.stage_wmapes]) / 100))
        for r, g in whole.groupby("round_id")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path,
                        default=Path("Experiments/Protocol_test/exp_records/binary_frozen"))
    parser.add_argument("--out", type=Path,
                        default=Path("Experiments/Protocol_test/analysis_plots/binary_frozen/rd_rounds.png"))
    parser.add_argument("--clients", type=int, default=30)
    args = parser.parse_args()

    combos = [c for c in ORDER if (args.records / c).exists()]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), squeeze=False)
    total = 0
    for ax, combo in zip(axes.ravel(), combos):
        for codec, (marker, column, filled) in SERIES.items():
            for csv in sorted((args.records / combo / "cases" / codec).glob("*/compression_records.csv")):
                points = dots(csv, column, args.clients)
                if not points:
                    continue
                color = COLOR.get(points[0][0], "#5B6675")
                ax.scatter([r for _, r, _ in points], [d for _, _, d in points], marker=marker, s=52,
                           facecolors=color if filled else "none", edgecolors=color,
                           linewidths=1.4, alpha=.85 if filled else 1.0, zorder=3 if filled else 2)
                total += len(points)
        ax.set_title(combo, fontsize=10.5)
        ax.set_xlabel("rate (bit/coordinate)", fontsize=9)
        ax.set_ylabel("WMAPE (dB rel. 100%)", fontsize=9)
        ax.grid(alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(combos):]:
        ax.axis("off")

    key = [
        ("o", "none", "#5B6675", "T  (own past 5)"),
        ("^", "none", "#5B6675", "O  (30 clients x 5)"),
        ("s", "none", "#5B6675", "hollow = retrained prior"),
        ("s", "#5B6675", "#5B6675", "filled = on the wire"),
        ("s", COLOR[5], COLOR[5], "trained @5"),
        ("s", COLOR[12], COLOR[12], "trained @12"),
    ]
    handles = [plt.Line2D([], [], marker=m, ls="", mfc=fc, mec=ec, ms=9, mew=1.4, label=l)
               for m, fc, ec, l in key]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("one dot = one codec at one round, averaged over 30 clients", y=.895, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, .86])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"{total} dots -> {args.out}")


if __name__ == "__main__":
    main()
