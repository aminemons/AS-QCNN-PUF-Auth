"""
PUF Dataset — PyTorch Dataset wrapper for Arbiter PUF CRPs.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PUFDataset(Dataset):
    """Dataset for Challenge-Response Pairs (CRPs) from Arbiter PUF simulation."""

    def __init__(self, challenges: np.ndarray, responses: np.ndarray, normalize: bool = True):
        """
        Args:
            challenges: np.ndarray [N, n_bits] in {-1, +1}
            responses:  np.ndarray [N] in {0, 1}
            normalize:  scale challenges to [-pi, pi] for angle embedding
        """
        if normalize:
            challenges = challenges.astype(np.float32) * np.pi
        else:
            challenges = challenges.astype(np.float32)
        self.challenges = torch.from_numpy(challenges)
        self.responses = torch.from_numpy(responses.astype(np.float32))

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        return self.challenges[idx], self.responses[idx]


def load_puf_dataloaders(
    npz_path: str,
    batch_size: int = 4096,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
):
    """Load a .npz CRP file and return train/val/test DataLoaders."""
    data = np.load(npz_path)
    challenges = data["challenges"]
    responses = data["responses"]

    dataset = PUFDataset(challenges, responses)
    n = len(dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    logger.info(f"Dataset: {n_train} train | {n_val} val | {n_test} test — batch {batch_size}")
    return train_loader, val_loader, test_loader
