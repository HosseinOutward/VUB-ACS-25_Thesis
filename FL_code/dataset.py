from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import SVHN, CIFAR10

if TYPE_CHECKING:
    from FL_code.run_fl import FLConfig


class SharedTensorDataset(Dataset):
    """Zero-copy dataset backed by shared-memory tensors."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X, self.y = X, y

        # Ensure tensors are in shared memory
        assert self.y.is_shared()
        assert self.X.is_shared()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# Dataset configurations: (mean, std, data_loader_func)
DATASET_CONFIG = {
    'SVHN': {
        'mean': [0.4377, 0.4438, 0.4728],
        'std': [0.1980, 0.2010, 0.1970],
    },
    'CIFAR10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616],
    },
    'SYNTHETIC': {
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0],
    },
}


def precompute_dataset_to_shared(
    dataset_name: str,
    data_folder: Path,
    split: str,
    dtype: torch.dtype = torch.float32,
    fraction: float | None = None,
    seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute dataset preprocessing and store in shared memory."""
    dataset_name = dataset_name.upper()
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG.keys())}")

    is_train = (split == "train")
    cfg = DATASET_CONFIG[dataset_name]
    data_path = Path(data_folder)

    # Load dataset
    if dataset_name == 'SYNTHETIC':
        sample_count = 2_000 if is_train else 800
        generator = torch.Generator().manual_seed((seed or 0) + (0 if is_train else 1))
        X = torch.randn(sample_count, 3, 32, 32, generator=generator)
        y = torch.arange(sample_count, dtype=torch.long) % 10
    elif dataset_name == 'SVHN':
        ds = SVHN(root=data_path / "SVHN", split=split, download=False)
        X = torch.from_numpy(ds.data).float().div_(255.0)
        y = torch.tensor(ds.labels, dtype=torch.long)
    else:  # CIFAR10
        ds = CIFAR10(root=data_path / "CIFAR10", train=is_train, download=True)
        X = torch.from_numpy(ds.data).float().permute(0, 3, 1, 2).div_(255.0)
        y = torch.tensor(ds.targets, dtype=torch.long)

    # Train is always shuffled; test only needs shuffling when a fraction is
    # taken, so the subset is not biased by the original file order.
    if is_train or fraction is not None:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        perm = torch.randperm(len(y), generator=generator)
        X, y = X[perm], y[perm]
        if fraction is not None:
            # Clone so share_memory_ ships only the subset, not the full-size storage
            # the [:n] view would otherwise keep alive.
            n = int(len(y) * fraction)
            X, y = X[:n].clone(), y[:n].clone()

    # Normalize in place to avoid transient full-size copies of the dataset
    mean = torch.tensor(cfg['mean']).view(1, 3, 1, 1)
    std = torch.tensor(cfg['std']).view(1, 3, 1, 1)
    X = X.sub_(mean).div_(std).to(dtype)

    X.share_memory_()
    y.share_memory_()
    return X, y


def create_dataloader(
    X: torch.Tensor, y: torch.Tensor, cfg: Any, device: torch.device,
    is_train: bool, client_id: int | None = None, num_clients: int | None = None
) -> DataLoader:
    """Create a train or test dataloader over shared tensors."""
    assert (client_id is None and is_train is False) or (client_id is not None and is_train is True)

    dataset = SharedTensorDataset(X, y)

    if is_train:
        # Training: IID partition across clients
        assert client_id is not None and num_clients is not None, "client_id and num_clients required for training"
        perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(cfg.seed))
        indices = perm[client_id::num_clients].tolist()
        dataset = Subset(dataset, indices)

        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            pin_memory=(device.type == "cuda"),
            num_workers=cfg.num_loader_workers,
            prefetch_factor=2 if cfg.num_loader_workers > 0 else None,
            persistent_workers=cfg.num_loader_workers > 0,
            shuffle=True,
        )
    else:
        # Testing: use full dataset, no shuffle, larger batch
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            num_workers=0
        )


def _force_class_coverage(y_train: torch.Tensor, y_test: torch.Tensor, cfg: FLConfig) -> None:
    """Brute-force label flips so fractioned debug splits still cover every class.

    Subsetting via dataset_fraction can drop classes from the test split or from a
    client's train partition, which breaks the AUC metrics downstream. This flips a
    few labels of the most frequent class to the missing ones — deliberately wrong
    labels, acceptable only for fraction-reduced debug runs.
    """
    assert cfg.num_classes is not None
    expected_classes = set(range(cfg.num_classes))

    def flip_missing(y: torch.Tensor, positions: torch.Tensor, split_name: str) -> None:
        labels = y[positions]
        missing = sorted(expected_classes - set(torch.unique(labels).tolist()))
        if not missing:
            return
        donor_class = int(torch.bincount(labels, minlength=cfg.num_classes).argmax())
        donor_positions = positions[labels == donor_class]
        assert len(donor_positions) > len(missing), (
            f"Cannot repair class coverage for {split_name}: {len(donor_positions)} samples of "
            f"donor class {donor_class} cannot cover {len(missing)} missing classes."
        )
        for donor_pos, missing_class in zip(donor_positions.tolist(), missing):
            y[donor_pos] = missing_class
        print(f"[Debug] Flipped {len(missing)} label(s) in {split_name} to cover missing classes {missing}.")

    flip_missing(y_test, torch.arange(len(y_test)), "test split")
    perm = torch.randperm(len(y_train), generator=torch.Generator().manual_seed(cfg.seed))
    for client_id in range(cfg.num_clients):
        flip_missing(y_train, perm[client_id::cfg.num_clients], f"client {client_id} train partition")
