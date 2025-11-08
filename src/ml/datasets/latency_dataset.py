"""PyTorch datasets for latency prediction."""

import torch
from torch.utils.data import Dataset
import numpy as np

from src.core.logging_config import get_logger

logger = get_logger()


class LatencyDataset(Dataset):
    """Dataset for latency prediction with sequence modeling."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 50):
        self.sequence_length = sequence_length

        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

        self.features = torch.nan_to_num(self.features, nan=0.0, posinf=1.0, neginf=-1.0)
        self.targets = torch.nan_to_num(self.targets, nan=1000.0, posinf=10000.0, neginf=50.0)

        self.targets = torch.clamp(self.targets, min=50.0, max=50000.0)

        self.feature_mean = self.features.mean(dim=0)
        self.feature_std = self.features.std(dim=0) + 1e-6

        mask = self.feature_std > 1e-5
        self.features[:, mask] = (
            self.features[:, mask] - self.feature_mean[mask]
        ) / self.feature_std[mask]

        self.targets = torch.log(self.targets)
        self.target_mean = self.targets.mean()
        self.target_std = self.targets.std() + 1e-6
        self.targets = (self.targets - self.target_mean) / self.target_std

        logger.verbose(
            "Dataset created",
            samples=len(self.features),
            feature_range=f"[{self.features.min():.3f}, {self.features.max():.3f}]",
            target_range=f"[{self.targets.min():.3f}, {self.targets.max():.3f}]",
        )

    def __len__(self) -> int:
        return max(0, len(self.features) - self.sequence_length)

    def __getitem__(self, idx: int):
        """Get sequence of features and target."""
        seq_features = self.features[idx : idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length]
        return seq_features, target

    def denormalize_prediction(self, pred: torch.Tensor) -> torch.Tensor:
        """Convert normalized prediction back to microseconds."""
        pred = pred * self.target_std + self.target_mean
        pred = torch.exp(pred)
        pred = torch.clamp(pred, min=50.0, max=50000.0)
        return pred


class StreamingLatencyDataset:
    """Streaming dataset for online learning."""

    def __init__(self, max_size: int = 10000, sequence_length: int = 50):
        self.max_size = max_size
        self.sequence_length = sequence_length

        self.features = []
        self.targets = []

    def add_sample(self, features: np.ndarray, target: float):
        """Add new training sample."""
        self.features.append(features)
        self.targets.append(target)

        if len(self.features) > self.max_size:
            self.features.pop(0)
            self.targets.pop(0)

    def get_dataset(self) -> LatencyDataset:
        """Convert to PyTorch dataset."""
        if len(self.features) < self.sequence_length + 1:
            return None

        features_array = np.array(self.features)
        targets_array = np.array(self.targets)

        return LatencyDataset(features_array, targets_array, self.sequence_length)

    def __len__(self) -> int:
        return len(self.features)
