from __future__ import annotations
import os
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import SVHN


class SharedTensorDataset(Dataset):
    """Zero-copy dataset backed by shared-memory tensors."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X, self.y = X, y

        # Ensure tensors are in shared memory
        assert self.y.is_shared()
        assert self.X.is_shared()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def precompute_svhn_to_shared(
    data_folder: str,
    split: str,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute SVHN preprocessing and store in shared memory."""
    ds = SVHN(root=os.path.join(data_folder, "SVHN"), split=split, download=False, transform=None)

    X = torch.from_numpy(ds.data).float().div_(255.0)
    y = torch.tensor(ds.labels, dtype=torch.long)

    mean = torch.tensor([0.4377, 0.4438, 0.4728]).view(1, 3, 1, 1)
    std = torch.tensor([0.1980, 0.2010, 0.1970]).view(1, 3, 1, 1)
    X = ((X - mean) / std).to(dtype)

    # Share memory to avoid duplication across processes
    X.share_memory_()
    y.share_memory_()

    return X, y


def create_dataloader(
    X: torch.Tensor, y: torch.Tensor, cfg, device: torch.device,
    is_train: bool, client_id: Optional[int] = None, num_clients: Optional[int] = None
) -> DataLoader:
    """
    Create dataloader for training or testing.

    Args:
        X: Input data tensor
        y: Label tensor
        cfg: Configuration object
        device: Device for pin_memory optimization
        is_train: True for training (with IID partition), False for testing
        client_id: Client ID for IID partition (only needed if is_train=True)
        num_clients: Total number of clients (only needed if is_train=True)
    """

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
            shuffle=True,
            pin_memory=device.type == "cuda",
            num_workers=cfg.num_loader_workers,
            persistent_workers=cfg.num_loader_workers > 0
        )
    else:
        # Testing: use full dataset, no shuffle, larger batch
        return DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
            pin_memory=device.type == "cuda",
            num_workers=0
        )
