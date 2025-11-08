"""Neural network models for latency prediction."""

import torch
import torch.nn as nn
from typing import Tuple

from src.core.logging_config import get_logger

logger = get_logger()


class LSTMLatencyModel(nn.Module):
    """LSTM model for latency prediction with attention mechanism."""

    def __init__(
        self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self._init_lstm_weights()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_lstm_weights(self):
        """Initialize LSTM weights for stable training."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)

    def _init_weights(self):
        """Initialize FC and attention weights."""
        for module in [self.attention, self.fc, self.confidence_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with prediction and confidence."""
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)

        if torch.isnan(lstm_out).any():
            logger.warning("NaN in LSTM output")
            lstm_out = torch.zeros_like(lstm_out)

        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        prediction = self.fc(context)
        confidence = self.confidence_head(context)

        prediction = torch.nan_to_num(prediction, nan=0.0)
        confidence = torch.nan_to_num(confidence, nan=0.5)

        return prediction, confidence


class GRULatencyModel(nn.Module):
    """GRU variant for latency prediction."""

    def __init__(
        self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with prediction and confidence."""
        x = self.input_norm(x)
        gru_out, hidden = self.gru(x)

        last_hidden = hidden[-1]
        prediction = self.fc(last_hidden)
        confidence = self.confidence_head(last_hidden)

        return prediction, confidence


class TransformerLatencyModel(nn.Module):
    """Transformer-based latency prediction model."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.input_proj(x)

        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]

        transformer_out = self.transformer(x)
        pooled = transformer_out.mean(dim=1)

        prediction = self.fc(pooled)
        confidence = self.confidence_head(pooled)

        return prediction, confidence
