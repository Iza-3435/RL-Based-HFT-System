"""Model training for latency prediction."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np

from src.core.logging_config import get_logger
from src.ml.models.latency_models import LSTMLatencyModel, GRULatencyModel
from src.ml.datasets.latency_dataset import LatencyDataset

logger = get_logger()


class LatencyModelTrainer:
    """Trains latency prediction models."""

    def __init__(
        self,
        model_type: str = "lstm",
        feature_size: int = 45,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        self.model_type = model_type
        self.feature_size = feature_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "lstm":
            self.model = LSTMLatencyModel(feature_size, hidden_size, num_layers, dropout)
        elif model_type == "gru":
            self.model = GRULatencyModel(feature_size, hidden_size, num_layers, dropout)
        else:
            self.model = LSTMLatencyModel(feature_size, hidden_size, num_layers, dropout)

        self.model = self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5
        )

        logger.verbose(f"Model trainer initialized", model_type=model_type, device=str(self.device))

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> Dict:
        """Train model on data."""
        try:
            dataset = LatencyDataset(features, targets)

            split_idx = int(len(dataset) * (1 - validation_split))
            train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
            val_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            best_val_loss = float("inf")
            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                train_loss = self._train_epoch(train_loader)
                val_loss = self._validate(val_loader, dataset)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if (epoch + 1) % 10 == 0:
                    logger.verbose(
                        f"Epoch {epoch+1}/{epochs}",
                        train_loss=f"{train_loss:.4f}",
                        val_loss=f"{val_loss:.4f}",
                    )

            logger.info(f"Training complete", best_val_loss=f"{best_val_loss:.4f}")

            return {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _train_epoch(self, loader: DataLoader) -> float:
        """Train single epoch."""
        self.model.train()
        total_loss = 0.0
        criterion = nn.MSELoss()

        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            predictions, _ = self.model(features)
            loss = criterion(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate(self, loader: DataLoader, dataset: LatencyDataset) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                predictions, _ = self.model(features)
                loss = criterion(predictions, targets)
                total_loss += loss.item()

        return total_loss / len(loader)

    def save_model(self, path: str):
        """Save trained model."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_type": self.model_type,
            },
            path,
        )
        logger.verbose(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.eval()
        logger.verbose(f"Model loaded from {path}")
