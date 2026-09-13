"""Rate-distortion dots for the matched T+rANS vs O+SW coder runs.

Two figures, both plotting only what the records hold: `rd_dots.png` puts one dot per codec per
round from the single-round cases, and `rd_frozen.png` puts one dot per codec per training round
from the frozen-reuse cases, with each round in the reuse window drawn behind it as a shadow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODECS = {"T_rans_retrained": ("T + rANS", "o"), "O_oracle_sw_retrained": ("O + SW", "^")}
COLOR = {4: "#7A3E9D", 5: "#1B6AA8", 6: "#2E8B57", 8: "#C1440E", 9: "#8E3B5A", 12: "#B4711A"}
ORDER = ["r56_femnist_nat", "r56_femnist", "r56_svhn", "r56_cifar10",
         "r18_eurosat64", "r18_imagenette160"]
MISSING_COLOR = "#5B6675"


def wmape_db(frame: pd.DataFrame) -> float:
    """Mean first-stage WMAPE over the frame's rows, in dB relative to 100%."""
    return 20 * np.log10(np.mean([eval(w)[0] for w in frame.stage_wmapes]) / 100)


def case_windows(root: Path, combo: str, codec: str,
                 clients: int = 30) -> list[tuple[int, float, float, list]]:
    """One entry per case: its start round, pooled rate and WMAPE, and per-round points.

    A round is included once every client has reported it, so a case still running contributes the
    rounds it has finished rather than a mean over whichever clients happen to be written.
    """
    windows = []
    for csv in sorted((root / combo / "cases" / codec).glob("*/compression_records.csv")):
        frame = pd.read_csv(csv)
        whole = frame.groupby("round_id").filter(lambda g: len(g) == clients)
        if whole.empty:
            continue
        rounds = sorted((int(r), g.coder_rate.mean(), wmape_db(g)) for r, g in whole.groupby("round_id"))
        windows.append((rounds[0][0], whole.coder_rate.mean(), wmape_db(whole), rounds))
    return sorted(windows)


def draw(root: Path, out: Path, shadows: bool, title: str) -> int:
    combos = [c for c in ORDER if (root / c).exists()]
    if not combos:
        return 0
    cols = min(3, len(combos))
    rows = -(-len(combos) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    axes = axes.ravel()
    plotted = 0
    for ax, combo in zip(axes, combos):
        for codec, (_, marker) in CODECS.items():
            for start, rate, dist, rounds in case_windows(root, combo, codec):
                color = COLOR.get(start, MISSING_COLOR)
                if shadows:
                    xs, ys = [r for _, r, _ in rounds], [d for _, _, d in rounds]
                    ax.plot(xs, ys, color=color, alpha=.35, lw=.9, zorder=1)
                    ax.scatter(xs, ys, marker=marker, s=34, color=color, alpha=.4, zorder=2)
                    # The training round is the window's first; ring it so reuse drift reads.
                    ax.scatter(xs[:1], ys[:1], marker=marker, s=34, facecolors="none",
                               edgecolors=color, linewidths=1.4, zorder=2)
                ax.scatter(rate, dist, marker=marker, s=115, color=color,
                           edgecolors="white", linewidths=.9, zorder=3)
                plotted += 1
        ax.set_title(combo, fontsize=10.5)
        ax.set_xlabel("rate on the wire (bit/coordinate)", fontsize=8.5)
        ax.set_ylabel("WMAPE (dB rel. 100%)", fontsize=8.5)
        ax.grid(alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes[len(combos):]:
        ax.axis("off")
    starts = sorted({s for c in combos for k in CODECS for s, _, _, _ in case_windows(root, c, k)})
    handles = [plt.Line2D([], [], marker=m, ls="", color=MISSING_COLOR, ms=10, label=l)
               for l, m in CODECS.values()]
    handles += [plt.Line2D([], [], marker="s", ls="", color=COLOR.get(s, MISSING_COLOR), ms=10,
                           label=f"round {s}") for s in starts]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9.5)
    fig.suptitle(title, y=.93, fontsize=9.5, linespacing=1.4)
    fig.tight_layout(rect=[0, 0, 1, .87 if "\n" in title else .90])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return plotted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("Experiments/Protocol_test/exp_records"))
    parser.add_argument("--out", type=Path, default=Path("Experiments/Protocol_test/analysis_plots"))
    parser.add_argument("--clients", type=int, default=30,
                        help="Rounds are plotted only once this many clients have reported them.")
    args = parser.parse_args()

    for root_name, out_name, shadows, title in (
        ("binary_combos", "rd_dots.png", False,
         "dot = one codec at one round, averaged over 30 clients"),
        ("binary_frozen", "rd_frozen.png", True,
         "dot = reuse window averaged over 30 clients\n"
         "shadow = one round (ringed = the training round)"),
    ):
        root = args.records / root_name
        n = draw(root, args.out / root_name / out_name, shadows, title)
        print(f"{root_name}: {n} dots -> {args.out / root_name / out_name}" if n
              else f"{root_name}: nothing completed yet")
        for combo in (c for c in ORDER if (root / c).exists()):
            for codec, (label, _) in CODECS.items():
                for start, rate, dist, rounds in case_windows(root, combo, codec):
                    span = f"r{start}..{rounds[-1][0]}" if len(rounds) > 1 else f"r{start}"
                    print(f"  {combo:20s} {label:9s} {span:8s} rate {rate:.4f}  {dist:+.2f} dB"
                          + ("".join(f"\n      round {r:2d}  rate {x:.4f}  {y:+.2f} dB"
                                     for r, x, y in rounds) if len(rounds) > 1 else ""))


if __name__ == "__main__":
    main()
