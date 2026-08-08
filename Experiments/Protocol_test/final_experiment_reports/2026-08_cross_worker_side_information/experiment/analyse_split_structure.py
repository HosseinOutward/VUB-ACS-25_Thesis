"""What kind of coordinate ends up on each of the three quantizers?

The reassignment gave every coordinate to whichever quantizer actually codes it best, so the
resulting labels are a measurement, not a design. This asks whether that measurement lines up with
anything structural - a layer, a neuron's incoming weights - or statistical, and whether it agrees
between workers. A partition that clusters structurally could be assigned cheaply for free; one
that does not is per-coordinate information somebody has to transmit.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.FL_core.models import initialize_model  # noqa: E402
from FL_code.FL_core.tensor_storage import load_tensor  # noqa: E402
from FL_code.FL_core.utils import StateDictManager  # noqa: E402
from FL_code.cancer_protocol.wz_quantizer import normalization_params  # noqa: E402
from FL_code.run_fl import FLConfig  # noqa: E402

PARTS = 3


def layout(fl_config: Path) -> StateDictManager:
    cfg = FLConfig.model_validate(
        json.loads(fl_config.read_text())
        | {"protocol": "wz_cancer", "run_name": "x", "records_dir": "/tmp", "seed": 43})
    return StateDictManager(initialize_model(cfg, torch.device("cpu")))


def purity(labels: NDArray[np.uint8], groups: int) -> float:
    """Mean share of a group held by its most common label, for equally sized consecutive groups.

    One is a perfectly clustered partition, one third is a partition unrelated to the grouping.
    """
    blocks = labels.reshape(groups, -1)
    counts = np.stack([(blocks == part).sum(axis=1) for part in range(PARTS)])
    return float((counts.max(axis=0) / blocks.shape[1]).mean())


def neuron_purity(
    labels: NDArray[np.uint8], manager: StateDictManager, rng: np.random.Generator
) -> pd.DataFrame:
    """How concentrated the labels are within each unit's incoming weights, against a shuffle.

    A conv or linear weight is stored output-unit major, so reshaping to (units, incoming) groups
    exactly the weights feeding one neuron or filter. The shuffled column is the same tensor's
    labels permuted, which is what purity would be if the assignment ignored structure entirely.
    """
    rows = []
    for key, shape, sl in zip(manager.keys, manager.shapes, manager.get_slices()):
        if len(shape) < 2:
            continue
        block = labels[sl]
        units = shape[0]
        rows.append({
            "parameter": key,
            "units": units,
            "incoming_per_unit": block.size // units,
            "purity": purity(block, units),
            "purity_shuffled": purity(rng.permutation(block), units),
        })
    return pd.DataFrame(rows)


def decile_labels(values: NDArray[np.float32]) -> NDArray[np.int32]:
    order = np.argsort(values, kind="stable")
    out = np.empty(values.size, dtype=np.int32)
    out[order] = np.arange(values.size) * 10 // values.size
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--deltas",
        type=Path,
        default=PROJECT_ROOT / "_/data/federated_learning/raw_client_deltas",
    )
    parser.add_argument("--round", type=int, default=20)
    parser.add_argument("--fl-config", type=Path, default=EXPERIMENT_DIR / "replay_fl_config.json")
    args = parser.parse_args()

    labels = np.load(args.labels)
    manager = layout(args.fl_config)
    slices, keys = manager.get_slices(), manager.keys
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)

    print("=== overall label shares per worker ===")
    for client_id, row in enumerate(labels):
        print(f"  worker {client_id}: " + "  ".join(
            f"part{part} {float((row == part).mean()):.3f}" for part in range(PARTS)))

    print("\n=== do workers agree on a coordinate's label? ===")
    agree = np.mean([
        float((labels[a] == labels[b]).mean())
        for a in range(len(labels)) for b in range(a + 1, len(labels))])
    print(f"  mean pairwise label agreement: {agree:.4f}  (chance is about 0.33)")

    depth = np.array([int(m.group(1)) if (m := re.search(r"layer(\d)", k)) else 0 for k in keys])
    parameter_of = np.repeat(np.arange(len(keys)), manager.numels)
    print("\n=== by depth (worker 0) ===")
    rows = []
    for d in sorted(set(depth.tolist())):
        mask = np.isin(parameter_of, np.where(depth == d)[0])
        rows.append({"depth": f"layer{d}" if d else "stem/fc", "coords": int(mask.sum()),
                     **{f"part{p}": round(float((labels[0][mask] == p).mean()), 3) for p in range(PARTS)}})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== by incoming weights of a neuron (worker 0) ===")
    neurons = neuron_purity(labels[0], manager, rng)
    big = neurons[neurons.units * neurons.incoming_per_unit > 50_000]
    print(big[["parameter", "units", "incoming_per_unit", "purity", "purity_shuffled"]]
          .to_string(index=False))
    print(f"  weighted purity {np.average(big.purity, weights=big.units * big.incoming_per_unit):.4f}"
          f" vs shuffled {np.average(big.purity_shuffled, weights=big.units * big.incoming_per_unit):.4f}")

    delta = load_tensor(args.deltas / f"round_{args.round}_client_0.pt")
    previous = load_tensor(args.deltas / f"round_{args.round - 1}_client_0.pt")
    normalized, normalized_prev = delta.clone(), previous.clone()
    for sl in slices:
        scale, center = normalization_params(normalized[sl])
        normalized[sl] = (normalized[sl] - center) / scale
        normalized_prev[sl] = (normalized_prev[sl] - center) / scale
    stats = {
        "|x| decile": decile_labels(np.abs(normalized.numpy())),
        "|x - prev| decile": decile_labels(np.abs((normalized - normalized_prev).numpy())),
    }
    for name, grouping in stats.items():
        print(f"\n=== by {name} (worker 0) ===")
        rows = [{name: d, **{f"part{p}": round(float((labels[0][grouping == d] == p).mean()), 3)
                             for p in range(PARTS)}} for d in range(10)]
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
