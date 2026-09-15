"""Frozen study: last ran against commit 3df7927; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout.

Measure how well cached decoder-side history predicts the next client-delta sign."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DELTAS: Final[Path] = PROJECT_ROOT / "_/data/federated_learning/raw_client_deltas"
DEFAULT_OUTPUT: Final[Path] = Path(__file__).resolve().parents[1] / "results/sign_predictability.csv"
FILE_NAME: Final[re.Pattern[str]] = re.compile(r"round_(\d+)_client_(\d+)\.pt")
LABELS: Final[NDArray[np.int8]] = np.array((-1, 0, 1), dtype=np.int8)


@dataclass(frozen=True, slots=True)
class DeltaSet:
    """A rectangular cached-delta index discovered by the experiment runner."""

    directory: Path
    paths: dict[tuple[int, int], Path]
    rounds: tuple[int, ...]
    clients: tuple[int, ...]

    @classmethod
    def discover(cls, directory: Path) -> DeltaSet:
        """Index cached tensors and require complete round-client coverage."""
        paths = {
            (int(match[1]), int(match[2])): path
            for path in directory.glob("round_*_client_*.pt")
            if (match := FILE_NAME.fullmatch(path.name))
        }
        assert paths, f"No round_<round>_client_<client>.pt tensors found in {directory}."
        rounds = tuple(sorted({round_id for round_id, _ in paths}))
        clients = tuple(sorted({client_id for _, client_id in paths}))
        missing = [
            (round_id, client_id)
            for round_id in rounds
            for client_id in clients
            if (round_id, client_id) not in paths
        ]
        assert not missing, f"Cached deltas are not rectangular; first missing entries: {missing[:5]}."
        return cls(directory, paths, rounds, clients)

    def load_sample(
        self, round_id: int, client_id: int, positions: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        """Load selected flat positions from one CPU float32 delta tensor."""
        path = self.paths[(round_id, client_id)]
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        assert isinstance(tensor, torch.Tensor), f"Cached delta is not a tensor: {path}."
        assert tensor.dtype == torch.float32 and tensor.device.type == "cpu", (
            f"Expected a CPU float32 tensor in {path}; got {tensor.dtype} on {tensor.device}."
        )
        return tensor.flatten()[torch.from_numpy(positions)].numpy()


def parse_args() -> argparse.Namespace:
    """Parse the saved-delta range and lightweight classifier settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas", type=Path, default=DEFAULT_DELTAS)
    parser.add_argument("--start-round", type=int, required=True)
    parser.add_argument("--end-round", type=int, required=True)
    parser.add_argument("--history-rounds", type=int, default=3)
    parser.add_argument("--clients", type=int, nargs="*", help="Target clients; defaults to all.")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sampled_history(
    delta_set: DeltaSet,
    rounds: range,
    clients: tuple[int, ...],
    positions: NDArray[np.int64],
) -> dict[tuple[int, int], NDArray[np.float32]]:
    """Load each sampled round-client vector once for all classifier fits."""
    return {
        (round_id, client_id): delta_set.load_sample(round_id, client_id, positions)
        for round_id in rounds
        for client_id in clients
    }


def feature_matrix(
    samples: dict[tuple[int, int], NDArray[np.float32]],
    target_round: int,
    target_client: int,
    history_rounds: int,
    clients: tuple[int, ...],
    same_round_clients: tuple[int, ...] = (),
) -> NDArray[np.float32]:
    """Stack the history and already-decoded current-round columns available at the decoder."""
    columns = [
        samples[(round_id, client_id)]
        for round_id in range(target_round - history_rounds, target_round)
        for client_id in clients
    ]
    columns.extend(samples[(target_round, client_id)] for client_id in same_round_clients)
    return np.column_stack(columns)


def cross_entropy_bits(
    target: NDArray[np.int8], probabilities: NDArray[np.float64], classes: NDArray[np.int8]
) -> float:
    """Return mean symbol cross entropy in bits for the classifier's class order."""
    class_columns = {int(label): column for column, label in enumerate(classes)}
    missing = sorted(set(map(int, np.unique(target))) - set(class_columns))
    assert not missing, f"Test labels {missing} do not occur in the classifier's training split."
    selected = probabilities[np.arange(len(target)), [class_columns[int(label)] for label in target]]
    return float(-np.log2(np.clip(selected, np.finfo(np.float64).tiny, 1.0)).mean())


def score_predictions(
    target: NDArray[np.int8], prediction: NDArray[np.int8], conditional_bits: float, marginal_bits: float
) -> dict[str, float]:
    """Return comparable classification and conditional-rate metrics."""
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "cross_entropy_bits": conditional_bits,
        "bits_gain": marginal_bits - conditional_bits,
    }


def evaluate(
    features: NDArray[np.float32],
    target: NDArray[np.int8],
    train: NDArray[np.int64],
    test: NDArray[np.int64],
    max_iter: int,
    seed: int,
) -> dict[str, float]:
    """Fit one small boosting classifier and score held-out parameter positions."""
    assert len(np.unique(target[train])) > 1, "Training labels contain only one delta-sign class."
    classifier = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=seed,
    ).fit(features[train], target[train])
    probabilities = classifier.predict_proba(features[test])
    counts = np.array([(target[train] == label).sum() for label in LABELS], dtype=np.float64)
    marginal = np.broadcast_to(counts / counts.sum(), (len(test), len(LABELS)))
    marginal_bits = cross_entropy_bits(target[test], marginal, LABELS)
    conditional_bits = cross_entropy_bits(target[test], probabilities, classifier.classes_)
    return score_predictions(
        target[test], classifier.predict(features[test]), conditional_bits, marginal_bits
    )


def main() -> None:
    """Compare own history with all decoder-available SI over the requested rounds and clients."""
    args = parse_args()
    assert 0 < args.test_fraction < 1
    assert args.history_rounds > 0 and args.samples > 1 and args.max_iter > 0
    assert args.start_round <= args.end_round

    delta_set = DeltaSet.discover(args.deltas)
    target_clients = tuple(args.clients) if args.clients else delta_set.clients
    assert target_clients and set(target_clients) <= set(delta_set.clients), (
        f"Requested clients {target_clients}; available clients are {delta_set.clients}."
    )
    required_rounds = range(args.start_round - args.history_rounds, args.end_round + 1)
    assert set(required_rounds) <= set(delta_set.rounds), (
        f"Rounds {required_rounds.start}..{required_rounds.stop - 1} are required; "
        f"available range is {delta_set.rounds[0]}..{delta_set.rounds[-1]}."
    )

    first = torch.load(delta_set.paths[(args.start_round, target_clients[0])], map_location="cpu", weights_only=True)
    assert isinstance(first, torch.Tensor)
    assert args.samples <= first.numel(), (
        f"Requested {args.samples} positions from deltas containing {first.numel()} values."
    )
    rng = np.random.default_rng(args.seed)
    positions = np.sort(rng.choice(first.numel(), args.samples, replace=False)).astype(np.int64)
    order = rng.permutation(args.samples)
    split = round(args.samples * (1 - args.test_fraction))
    assert 0 < split < args.samples, "test-fraction leaves an empty training or test split."
    train, test = order[:split], order[split:]
    samples = sampled_history(delta_set, required_rounds, delta_set.clients, positions)

    rows: list[dict[str, int | float | str]] = []
    for target_round in range(args.start_round, args.end_round + 1):
        for target_client in target_clients:
            target = np.sign(samples[(target_round, target_client)]).astype(np.int8)
            for model, clients, same_round_clients in (
                ("own_history", (target_client,), ()),
                ("decoder_si", delta_set.clients, tuple(c for c in delta_set.clients if c < target_client)),
            ):
                features = feature_matrix(
                    samples, target_round, target_client, args.history_rounds, clients,
                    same_round_clients,
                )
                metrics = evaluate(features, target, train, test, args.max_iter, args.seed)
                row = {
                    "round_id": target_round,
                    "client_id": target_client,
                    "model": model,
                    "history_rounds": args.history_rounds,
                    "n_si_clients": len(clients),
                    "n_same_round_clients": len(same_round_clients),
                    "n_features": features.shape[1],
                    "n_train": len(train),
                    "n_test": len(test),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"round {target_round}, client {target_client}, {model}: "
                    f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
                    f"bits_gain={metrics['bits_gain']:.4f}",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
